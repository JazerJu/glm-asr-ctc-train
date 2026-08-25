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

Code-switching + GigaSpeech (2026-08 轮新增的四个语料):

```bash
bash scripts/download/download_codeswitch.sh all      # 或 talcs / cs_dialogue / ascend / gigaspeech
```

- `download_codeswitch.sh` 下载 TALCS(587h)、CS-Dialogue(104h)、ASCEND(10.6h)、
  GigaSpeech M(1000h),产出的目录布局与 `prepare_manifests.py` 里对应的 builder 严格匹配。
  几个反直觉的点已经写在脚本注释里:TALCS 是 **model** repo 且分 22 卷;CS-Dialogue 要
  用 short_wav 且必须解到 `<root>/data/`;GigaSpeech 是 gated,需要先在网页同意条款。

Legacy/manual scripts:

- `download_all_legacy.sh` is the older one-shot downloader and does not handle the current credentialed/parallel workflow.
- `download_datasets_legacy.py` is an older Hugging Face dataset downloader for alternative datasets, not the current 10k-hour training set path.
- `prepare_wenetspeech_m_official.py` regenerates WenetSpeech M metadata from official metadata when needed.

The WenetSpeech M metadata archive is stored next to the downloader as `wenet_m_metadata.tar.zst`
(39MB,`.gitignore` 里对它开了例外——没有它 `download_wenetspeech_m_official.sh` 跑不起来,
而它又无法从公开源重新生成)。

## 凭据(不在仓库里,需要另行提供)

大部分语料可以直接重下,但这两个不行:

| 语料 | 需要什么 | 放到哪 |
|---|---|---|
| Common Voice | Mozilla Data Collective API key | `.secrets/mdc_api_key` 或 `MDC_API_KEY` |
| WenetSpeech M | 官方解压密码(向 WenetSpeech 作者申请) | `.secrets/wenetspeech_password` 或 `WENET_PASSWORD_FILE` |
| KsponSpeech / GigaSpeech | HF token(gated,需先同意条款) | `/data/.cache/huggingface/token` 或 `HF_TOKEN` |

`.secrets/` 被 `.gitignore` 排除,换机器时要手动带过去。
