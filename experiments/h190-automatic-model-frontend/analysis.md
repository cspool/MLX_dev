# H190 result: automatic FX and ONNX frontend

Run195 is supported with `audit_integrity=true` and 10/10 gates.

- Real PyTorch FX/ShapeProp and ONNX 1.22 ModelProto imports both execute.
- Each produces six legal canonical nodes; all six signatures match.
- Twelve source nodes retain source-op/shape to canonical-plan lineage.
- CDC, tag, PE, register/bank and aligned SPM/DMA plans are legal.
- Twelve KernelProfiles execute twice: 24/24 runs and 12/12 replay identities.

The canonical graph is extracted from executable model representations rather
than manually authored operator YAML. No paper performance target is consumed.
