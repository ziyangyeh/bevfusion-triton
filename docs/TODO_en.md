# TODO

- Add Triton benchmark scripts for the current operator set.
- Benchmark and tune launch and tile parameters such as `_BLOCK_SIZE`, `_CHANNEL_BLOCK`, `_BLOCK_N`, and `_BLOCK_C`.
- Sweep typical shapes and dtypes (`fp32`, `fp16`, `bf16`) before changing shared Triton block-size defaults.
- Add multi-GPU validation and test result aggregation for BEVFusion Lightning evaluation. Current nuScenes detection evaluation path assumes single-GPU evaluation.
