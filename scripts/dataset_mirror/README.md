# OpenSLR dataset mirrors

This workflow preserves the original AISHELL-1 SLR33 and MAGICDATA SLR68
archives under `/media/jju/ExtraDisk/ctc_train_data`, verifies their exact
HTTP sizes and gzip streams, writes SHA256 checksums, and uploads them to the
configured Hugging Face dataset repositories.

Run:

```bash
nohup bash scripts/dataset_mirror/run_mirror.sh \
  > /media/jju/ExtraDisk/ctc_train_data/_state/run.log 2>&1 &
```

Status:

```bash
bash scripts/dataset_mirror/status.sh
```

The Hugging Face CLI must already be authenticated. Authentication and upload
always use `https://huggingface.co`, even when the shell-wide `HF_ENDPOINT`
points to a download mirror. Override paths or repositories with:

```bash
export CTC_ARCHIVE_ROOT=/media/jju/ExtraDisk/ctc_train_data
export AISHELL_HF_REPO=JazerJu/aishell1-full-slr33
export MAGICDATA_HF_REPO=JazerJu/magicdata-slr68-raw
```
