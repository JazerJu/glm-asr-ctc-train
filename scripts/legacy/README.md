# Legacy Scripts

These files are preserved for reference but are not the current GLM-ASR DDP
training path.

- `train_qwen_asr_legacy.py`: older Qwen-ASR/AISHELL CTC experiment.
- `run_train_qwen_legacy.sh`: runner for the older external Qwen-ASR project.
- `run_warmup_train_ctc_legacy.sh`: older `train_ctc.py` warmup path.
- `sweep_blocks_legacy.sh`: older 8-GPU Phase2 sweep script with stale manifest defaults.

Current warmup/training entrypoint:

```bash
bash scripts/run_ddp.sh
```

Current Phase2 benchmark/selection scripts:

```bash
bash scripts/run_phase2_ops_benchmark.sh
bash scripts/run_phase2_auto_select.sh
```
