# Full catalogue link and post-graph readback: component verification

117 tests passed in 12.94 seconds. `results.xml` is included; generated fixture
inputs/binaries and running full-model results are not included in this commit.
Reproduce with the project environment:

```sh
python -m pytest tests/test_model_event_linker.py tests/test_event_model_windows.py tests/test_event_graph_plan.py tests/test_streaming_pair_events.py tests/test_model_readback_timing.py -q
```

Coverage includes complete small numerical graphs across the four backends,
compact linkage and original pairs/parents, independent mapping/work audits,
registered output-readback timing, the actual event-runner-to-readback CLI path,
and rejection of changed options. Constant-token fixtures are not model argmax
validation. Existing frozen C++ event and native cores were not changed.

Local full BERT linkage (not executed work): 3046 sources, 3838 windows,
693 pairs, 13136277 logical blocks, 11113202670 dynamic events. Independent
audit covers 785 distinct window mappings. Large generated inputs remain local:

- graph SHA256: `8544dfa5086d38ee230005e443cdd87c3d7f9ca2ceb71da9e3c50e3016e3c715`
- audit SHA256: `e4fd6abff7ff85ce4d4bf21b639de4ce86131e24acf04b85d182f927406224df`
- results.xml SHA256: `5f312f7240d155d14846673615d982116a56b38012556c0c214f556403c91b22`

The full BERT event process and the original native/Chipyard processes are still
running. No model performance error, full numerical acceptance, Chipyard
acceptance, GPU result or RTL/PPA result is certified by this component evidence.
