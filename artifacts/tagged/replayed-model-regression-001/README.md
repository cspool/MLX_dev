# Replay-backed Llama event compilation — component evidence

131 tests passed in 12.64 seconds. Reproduce with the project environment:

```sh
python -m pytest tests/test_predicate_closure.py tests/test_replayed_window_audit.py tests/test_model_event_linker.py tests/test_event_model_windows.py tests/test_event_graph_plan.py tests/test_streaming_pair_events.py tests/test_model_readback_timing.py -q
```

This commit includes source, documentation and test XML, not running model
outputs, checkpoint data or the large replay traces.

The actual local workflow additionally executed three RV64 argmax windows over
SHA-bound logits saved by the accepted original native Llama full run. It
captured 95,997 branch decisions and reproduced tokens 393/372/338. Three where
predicates were observed by executing the original asset-free 36-node
control/view dependency closure. These are two distinct replay evidence types,
not a new full numerical run and not paired or Chipyard acceptance.

All 12,133 windows were independently rebuilt, followed by full compact-link
mapping audit (6,181 sources, 873 pairs, 2,273,057 blocks, 68,045,439,147 declared
events). Original running event/Chipyard code was not modified.

Local artifact SHA256 anchors (large artifacts are not included):

- Window rebuild audit: `0ba89143441fb8fe5733d5b0c7ea4cf0555c95aa2f2f4af543631672897bf3f2`
- Full link audit: `3f961f77f76563428b177d6bb6787bdd2ab8ce7875cc8c36c5677afec449b02b`
- Linked graph: `327b59701c0f9a0124e19c899752565b9e490a65313bb1318f3d890e9646cc39`

The complete Llama and BERT event graphs are running separately. Their declared
event counts are not completed work or performance results. Final readback,
same-input comparison with completed concurrent numerical runs, actual address
and buffer-lifetime validation, and Chipyard full-model acceptance remain open.
