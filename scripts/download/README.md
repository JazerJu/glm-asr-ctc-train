# Download Scripts

Current entrypoint:

```bash
bash scripts/download/run_downloads_fast.sh
```

Active scripts used by that entrypoint:

- `run_downloads_fast.sh` starts tmux jobs for downloader, monitor, and WenetSpeech M.
- `download_remaining_fast.sh` downloads/resumes AISHELL-1, MAGICDATA, Common Voice, LibriSpeech, KsponSpeech, MLS, and optional WenetSpeech HF DEV data.
- `download_wenetspeech_m_official.sh` downloads/decrypts/extracts official WenetSpeech M packages.
- `download_common_voice_mdc.py` fetches Common Voice archives through Mozilla Data Collective.
- `download_wenetspeech_hf.py` handles WenetSpeech HF subsets for smoke/manual use.
- `monitor_downloads.sh`, `download_status.sh`, `monitor.sh`, and `status.sh` inspect active jobs and logs.

Legacy/manual scripts:

- `download_all_legacy.sh` is the older one-shot downloader and does not handle the current credentialed/parallel workflow.
- `download_datasets_legacy.py` is an older Hugging Face dataset downloader for alternative datasets, not the current 10k-hour training set path.
- `prepare_wenetspeech_m_official.py` regenerates WenetSpeech M metadata from official metadata when needed.

The WenetSpeech M metadata archive is stored next to the downloader as `wenet_m_metadata.tar.zst`.
