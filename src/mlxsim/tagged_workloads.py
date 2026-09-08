"""Small source-level MLX workloads for automatic native-simulator compilation.

Vectors carry independent batch lanes. SWA uses scalar-feature attention for
the first SIMD/4 lanes, matching the current transcendental-unit contract;
other lanes have zero inputs and zero observable outputs. These are mechanism
fixtures, not a claim of full-model or paper-scale performance reproduction.
"""

from __future__ import annotations

import math

import numpy as np


class Graph:
    def __init__(self, name, lanes, iterations):
        self.lanes = lanes
        self.document = {
            "schema_version": 2,
            "name": name,
            "iterations": iterations,
            "inputs": {},
            "operations": [],
            "outputs": [],
        }
        self.constants = {}
        self.current_layer = -1
        self.current_region = ""

    def new_layer(self, label):
        self.current_layer += 1
        self.region(label)

    def region(self, label):
        self.current_region = f"{self.current_layer}:{label}"

    def input(self, name, values):
        self.document["inputs"][name] = np.asarray(values, dtype=np.float16).tolist()
        return name

    def constant(self, value):
        value = float(np.float16(value))
        if value not in self.constants:
            name = f"constant_{len(self.constants)}"
            self.constants[value] = self.input(name, np.full(self.lanes, value))
        return self.constants[value]

    def op(self, operation, *inputs):
        name = f"v{len(self.document['operations'])}"
        self.document["operations"].append(
            {
                "id": name,
                "op": operation,
                "inputs": list(inputs),
                "region": self.current_region,
                "layer": self.current_layer,
            }
        )
        return name


def bsmm(graph, inputs):
    values = list(inputs)
    a, b, c, d = [graph.constant(v) for v in (1.0, 0.5, -0.5, 1.0)]
    for stage in range(int(math.log2(len(values)))):
        graph.new_layer(f"bsmm_stage_{stage}")
        stride = 1 << stage
        previous = values.copy()
        for base in range(0, len(values), 2 * stride):
            for offset in range(stride):
                lo, hi = base + offset, base + offset + stride
                graph.region(f"butterfly_{lo}_{hi}")
                values[lo] = graph.op("fma", previous[hi], b, graph.op("mul", previous[lo], a))
                values[hi] = graph.op("fma", previous[hi], d, graph.op("mul", previous[lo], c))
    return values


def bsmm_reference(inputs):
    result = np.asarray(inputs, dtype=np.float16).copy()
    for stage in range(int(math.log2(len(result)))):
        stride = 1 << stage
        old = result.copy()
        for base in range(0, len(result), 2 * stride):
            for offset in range(stride):
                lo, hi = base + offset, base + offset + stride
                result[lo] = (old[lo] + (old[hi] * np.float16(0.5)).astype(np.float16)).astype(
                    np.float16
                )
                result[hi] = ((old[lo] * np.float16(-0.5)).astype(np.float16) + old[hi]).astype(
                    np.float16
                )
    return result


def fft(graph, real, imaginary):
    n = len(real)
    bits = int(math.log2(n))
    order = [int(f"{index:0{bits}b}"[::-1], 2) for index in range(n)]
    real = [real[i] for i in order]
    imaginary = [imaginary[i] for i in order]
    negative = graph.constant(-1.0)
    for stage in range(1, bits + 1):
        graph.new_layer(f"fft_stage_{stage}")
        width = 1 << stage
        old_r, old_i = real.copy(), imaginary.copy()
        for base in range(0, n, width):
            for offset in range(width // 2):
                lo, hi = base + offset, base + offset + width // 2
                graph.region(f"complex_butterfly_{lo}_{hi}")
                angle = -2 * math.pi * offset / width
                cosine = graph.constant(round(math.cos(angle), 12))
                sine = graph.constant(round(math.sin(angle), 12))
                minus_sine = graph.constant(-round(math.sin(angle), 12))
                tr = graph.op("fma", old_i[hi], minus_sine, graph.op("mul", old_r[hi], cosine))
                ti = graph.op("fma", old_i[hi], cosine, graph.op("mul", old_r[hi], sine))
                real[lo], imaginary[lo] = (
                    graph.op("add", old_r[lo], tr),
                    graph.op("add", old_i[lo], ti),
                )
                real[hi] = graph.op("fma", tr, negative, old_r[lo])
                imaginary[hi] = graph.op("fma", ti, negative, old_i[lo])
    return real, imaginary


def fft_reference(real, imaginary):
    n = len(real)
    bits = int(math.log2(n))
    order = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)]
    real = np.asarray(real, dtype=np.float16)[order].copy()
    imaginary = np.asarray(imaginary, dtype=np.float16)[order].copy()
    for stage in range(1, bits + 1):
        width = 1 << stage
        old_r, old_i = real.copy(), imaginary.copy()
        for base in range(0, n, width):
            for offset in range(width // 2):
                lo, hi = base + offset, base + offset + width // 2
                angle = -2 * math.pi * offset / width
                cosine = np.float16(round(math.cos(angle), 12))
                sine = np.float16(round(math.sin(angle), 12))
                tr = (
                    (old_r[hi] * cosine).astype(np.float16) + (old_i[hi] * -sine).astype(np.float16)
                ).astype(np.float16)
                ti = (
                    (old_r[hi] * sine).astype(np.float16) + (old_i[hi] * cosine).astype(np.float16)
                ).astype(np.float16)
                real[lo], real[hi] = old_r[lo] + tr, old_r[lo] - tr
                imaginary[lo], imaginary[hi] = old_i[lo] + ti, old_i[lo] - ti
    return real, imaginary


def swa(graph, query, keys, values):
    graph.new_layer("scores")
    scores = []
    for index, key in enumerate(keys):
        graph.region(f"dot_{index}")
        scores.append(graph.op("mul", query, key))
    graph.new_layer("softmax")
    maximum = scores[0]
    for score in scores[1:]:
        maximum = graph.op("max", maximum, score)
    exponentials = [
        graph.op("exp", graph.op("fma", maximum, graph.constant(-1), score)) for score in scores
    ]
    denominator = exponentials[0]
    for value in exponentials[1:]:
        denominator = graph.op("add", denominator, value)
    probabilities = [graph.op("div", weight, denominator) for weight in exponentials]
    graph.new_layer("weighted_sum")
    result = graph.constant(0)
    for probability, value in zip(probabilities, values, strict=True):
        result = graph.op("fma", probability, value, result)
    return result


def swa_reference(query, keys, values):
    lanes = len(query)
    active = lanes // 4
    scores = (np.asarray(keys, dtype=np.float16)[:, :active] * query[:active]).astype(np.float16)
    maximum = np.max(scores, axis=0)
    exponentials = np.exp((scores - maximum).astype(np.float16).astype(np.float32)).astype(
        np.float16
    )
    denominator = exponentials[0].copy()
    for value in exponentials[1:]:
        denominator = (denominator + value).astype(np.float16)
    result = np.zeros(lanes, dtype=np.float16)
    for weight, value in zip(exponentials, values, strict=True):
        probability = (weight / denominator).astype(np.float16)
        product = (probability * value[:active]).astype(np.float16)
        result[:active] = (result[:active] + product).astype(np.float16)
    return result


def workload(name, lanes=32, iterations=2):
    if name not in ("bsmm", "fft_cmp", "swa", "transformer_block"):
        raise ValueError(f"unsupported workload {name}")
    graph = Graph(name, lanes, iterations)
    data = np.asarray(
        [np.linspace(-0.25 + row / 16, 0.25 + row / 16, lanes) for row in range(4)],
        dtype=np.float16,
    )
    if name in ("swa", "transformer_block"):
        data[:, lanes // 4 :] = 0
    inputs = [graph.input(f"input_{i}", vector) for i, vector in enumerate(data)]
    if name == "bsmm":
        outputs, expected = bsmm(graph, inputs), bsmm_reference(data)
    elif name == "fft_cmp":
        imaginary_data = np.asarray(data * np.float16(0.25), dtype=np.float16)
        imaginary = [graph.input(f"imag_{i}", vector) for i, vector in enumerate(imaginary_data)]
        real, imag = fft(graph, inputs, imaginary)
        expected_r, expected_i = fft_reference(data, imaginary_data)
        outputs, expected = real + imag, np.concatenate((expected_r, expected_i))
    elif name == "swa":
        outputs = [swa(graph, inputs[0], inputs[1:3], inputs[2:4])]
        expected = [swa_reference(data[0], data[1:3], data[2:4])]
    else:
        # A declared small composition, not a full Transformer implementation:
        # Fourier mixing -> butterfly projection -> scalar-feature SWA -> residual.
        real, _ = fft(graph, inputs, [graph.constant(0)] * 4)
        mixed, _ = fft_reference(data, np.zeros_like(data))
        projected, reference = bsmm(graph, real), bsmm_reference(mixed)
        attention = swa(graph, projected[0], projected[1:3], projected[2:4])
        graph.new_layer("residual")
        outputs = [graph.op("add", attention, inputs[0])]
        expected = [
            (swa_reference(reference[0], reference[1:3], reference[2:4]) + data[0]).astype(
                np.float16
            )
        ]
    graph.document["outputs"] = outputs
    golden = {
        name: tuple(int(v) for v in np.asarray(vector, dtype=np.float16).view(np.uint16))
        for name, vector in zip(outputs, expected, strict=True)
    }
    return graph.document, golden
