# Corrected H39 protocol: canonical-key reload audit

Run044 is immutable and inconclusive. It nevertheless proves that all five H38
checkpoints strict-load and reproduce all ten run043 metrics exactly. Its sole
failed integrity gate expected 50 legacy LayerNorm aliases, but H38's current
Transformers saver wrote canonical `.weight/.bias` names; all five checkpoints
therefore correctly report zero renamed keys.

Run045 binds the complete run044 report (12,279 bytes, SHA-256
`0629a3d8...3171`) and changes only
`expected_layernorm_alias_count: 50 -> 0`. All run043/config/script,
checkpoint, tokenizer, dataset, runtime, topology, density, strict-load,
evaluation, and 0.02-point metric gates remain identical to the original H39
protocol. The correction is serialization-format evidence, not a target or
metric adjustment.

Execute one complete five-setting reload audit. H39 is supported only if the
same ten full-validation metrics pass and every corrected integrity gate passes.
Do not alter tolerance, checkpoint bytes, or model topology after run045.
