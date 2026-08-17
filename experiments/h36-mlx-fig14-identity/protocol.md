# H36 protocol: explicit-identifier audit of the supplied Fig. 14 floorplan

## Question and hypothesis

The user-supplied primary Fig. 14 raster contains a clearly legible chip,
project, or architecture-family identifier—or at least two clearly legible
exact parent-hardware values—not preserved in the Markdown extraction. Such
text could refine H34/H35's access-limited origin result.

This is a text/label audit, not visual architecture matching. Floorplan shape,
block placement, colors, regular arrays, logos without readable names, or
resemblance to another chip score zero.

## Frozen source

Inspect only
`MLX Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures/_page_9_Picture_0.jpeg`,
a 16,309-byte 266x213 RGB JPEG with SHA-256
`dc77320cffffb71ae50b3e388677eeea71df34bcdab1d8781e1faf73498da2ba`.
Bind the complete paper and H35 run040. Do not inspect another raster, PDF
rendering, web image, or candidate floorplan.

## Inspection and observation schema

Use the local image viewer at original detail once. Record one YAML observation
at the config's frozen path containing:

- image path, bytes, SHA-256, dimensions, mode, and inspection time;
- a neutral description of visible regions;
- every **clearly legible** text string relevant to identity or the seven
  numeric H34 parent-hardware fields, each classified as `chip_identifier`,
  `project_or_family_identifier`, `process_node`, `frequency`, `simd_width`,
  `mesh_dimensions`, `peak_throughput`, `pe_array_area`, `pe_array_power`, or
  `generic_label`;
- candidate mapping only for exact visible tokens among `DFU-E`, `M2-DFU`,
  `DFGAS`, `SimICT`, `DSAGEN`, `Assassyn`, `DPU-s`, `HTC-3000`, and
  `HTC-3500`; and
- an explicit list of text that is too small or blurred to transcribe.

Only confidence `clear` may enter a gate. Do not guess characters, use layout
resemblance, enhance/generate missing detail, or infer a name from authorship.
Generic labels such as `PE`, `memory`, `NoC`, `MLX`, or `floorplan` do not count
as candidate/family identity.

## Decision gates

H36 is supported if at least one clear non-generic chip/project/family
identifier is visible, or if at least two clear exact numeric parent-hardware
values are visible. It is rejected if the image is inspectable but neither
condition holds, and inconclusive only if the frozen raster cannot be opened or
the observation/integrity schema fails.

An exact parent is supported only if the figure explicitly labels the full
taped-out design with one registered hardware candidate (`DFU-E`, `M2-DFU`,
`DPU-s`, `HTC-3000`, or `HTC-3500`). A simulator, method, framework, loose
project name, or numeric values alone do not suffice. Architecture-family
attribution still uses H34's separate gate and cannot arise from layout
resemblance.

## Stopping rule

Perform one original-detail visual pass and one schema-audit run. Do not upscale,
sharpen, OCR-sweep, or inspect other paper images after seeing the result. If no
clear identifier is present, close the supplied-figure origin route; only a new
author statement, primary full text, or artifact may reopen exact provenance.
