# Full-paper experiment inventory

This is the completion checklist. `target` means a paper anchor is captured with provenance; `runner` means an executable experiment exists; `pass` means every covered numeric point is within 10% or the item is correctly classified as non-reproducible from public evidence.

| Paper item | Experiment | Evidence type in paper | Required reproduction artifact | Target | Runner | Pass |
|---|---|---|---|---:|---:|---:|
| Fig. 2 | AGX Orin dense vs FFT normalized execution time and L2/L1 hit rate at N=512/8K | Native GPU profile | GPU profile manifest + replay/native results | partial | no | no |
| Fig. 3 | H100 roofline, OI/performance points, CUDA utilization and QKV+attention FLOPs | Native GPU profile + analytical roofline | GPU roofline runner and targets | partial | no | no |
| Fig. 5 | Dominant Q/K/V frequencies across Llama2-7B layers | Model activation measurement | Activation capture + spectrum runner | no | no | no |
| Fig. 6 | K frequency energy at Llama2 layers 1 and 16 | Model activation measurement | Spectrum runner and digitized curves | no | no | no |
| Table II / Fig. 14 | 12-nm area and power breakdown | RTL synthesis + silicon | Exact table replay, component power model, provenance limitation | yes | no | no |
| Fig. 15(a) | ViT dense/BSMM/FNet/MLX accuracy and compute | Training from scratch | Training recipe + FLOP model + result audit | partial | no | no |
| Fig. 15(b) | BERT layer-count sensitivity | Retraining | Training recipe + FLOP model + audit | partial | no | no |
| Fig. 15(c,d) | Llama2/InternLM2 LoRA accuracy/perplexity and compute | Fine-tuning/evaluation | Model recipes, cached targets, FLOP model | partial | no | no |
| Fig. 16 | Block-size B sensitivity on ViT/Llama2/InternLM2 | Training/fine-tuning | Parameter sweep runner | partial | no | no |
| Fig. 17 | H100 prefill/decode speedup vs eager/FA | Native GPU benchmark | Native/trace GPU runner + audit | partial | no | no |
| Fig. 18 | Latency, energy, and algorithm-normalized speedup vs sparse accelerators | MLX cycle simulation + cited baselines | Reduced-design simulator + baseline replay | yes | no | no |
| Table V / Fig. 19 | FPGA resources and FABNet latency breakdown | FPGA implementation/simulation | Resource replay + workload timing model | partial | no | no |
| Fig. 20 | Eight Llama2 kernels vs Xavier, dense and sparse GPU | Silicon/MLX + native GPU | Full-design timing/power + GPU targets | partial | no | no |
| Fig. 21 | Llama2 end-to-end speedup, GEMM share, and memory | Silicon/simulation + native GPU | End-to-end graph and capacity model | partial | no | no |
| Fig. 22 | PE pipeline utilization on BSMM and chunk FFT | Cycle simulation | Event-simulator utilization sweep | yes | yes | yes (digitized targets, max error 7.1%) |
| Fig. 23 | SIMD and mesh scalability | Cycle simulation | Architecture sweep | yes | yes | yes (exploratory calibration, max error 3.2%) |
| Fig. 24 | FFT/BSMM/SWA workload sweep vs Orin/RTX 3090 | MLX simulation + native GPU | Workload sweep with GPU baseline manifest | yes | no | no |
| Fig. 25 | Roofline utilization heatmaps | Derived from performance/traffic | Roofline audit | partial | no | no |

Conceptual figures 1, 4, 7-13 specify algorithms, architecture, or mappings rather than standalone numeric experiments. Their claims are covered by unit tests for FFT compression, hierarchical BSMM, CDC closure, routing, pipeline overlap, and dense/SWA mappings.
