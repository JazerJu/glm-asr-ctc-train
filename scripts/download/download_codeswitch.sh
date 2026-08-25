#!/usr/bin/env bash
# 中英混杂语料 + GigaSpeech M 下载器（2026-08 轮新增的四个语料）
#
# 产出的目录布局与 prepare_manifests.py 里的 build_talcs / build_cs_dialogue /
# build_ascend / build_gigaspeech 严格对应，下完直接能跑 manifest 生成。
#
#   用法：bash scripts/download/download_codeswitch.sh [talcs|cs_dialogue|ascend|gigaspeech|all]
#   数据根目录：DATA_DIR（默认 /data/datasets）
#
# 注意事项（都是实际踩过的，别想当然）：
#   * TALCS 在 HF 上是 **model** repo，不是 dataset，且是 22 个分卷 tar，要 cat 拼接。
#   * CS-Dialogue 用 short_wav（句级，均长 9.62s），不用 long_wav（整场 ~50min，
#     需要靠 TextGrid 切分）。压缩包自带 "short_wav/" 顶层目录，必须解到 <root>/data/
#     才能让 wav.scp 里的相对路径对得上。
#   * GigaSpeech 是 gated=auto：要先在网页上同意条款，再用带 token 的账号下载。
#   * ASCEND / GigaSpeech 的音频内嵌在 parquet 里，由 prepare_manifests.py 解出来，
#     这个脚本只负责把 parquet 拉下来。
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TARGET="${1:-all}"

# HF token：GigaSpeech 必需，其余可选
for p in "${HF_TOKEN_FILE:-}" /data/.cache/huggingface/token ~/.cache/huggingface/token; do
  [ -n "$p" ] && [ -f "$p" ] && { export HF_TOKEN="$(cat "$p")"; break; }
done

say() { echo "[$(date '+%F %T')] $*"; }
need() { command -v "$1" >/dev/null || { echo "缺少命令：$1" >&2; exit 1; }; }
need hf

# hf download 自带断点续传，重跑即续传
dl() {  # dl <repo> <repo_type> <local_dir> [include_pattern...]
  local repo="$1" rtype="$2" dest="$3"; shift 3
  local args=(download "$repo" --repo-type "$rtype" --local-dir "$dest")
  for pat in "$@"; do args+=(--include "$pat"); done
  say "hf ${args[*]}"
  hf "${args[@]}"
}

do_talcs() {
  local root="$DATA_DIR/talcs"
  say "TALCS (587h 中英混杂, TAL Education) -> $root"
  dl csukuangfj/tal_csasr model "$root" "TAL_CSASR.tar.part*"
  if [ ! -d "$root/TALCS_corpus" ]; then
    say "拼接 22 个分卷并解包（约 587h wav，需要充足磁盘）"
    cat "$root"/TAL_CSASR.tar.part?? | tar xf - -C "$root"
  else
    say "TALCS_corpus/ 已存在，跳过解包"
  fi
  say "TALCS 完成：$(du -sh "$root" 2>/dev/null | cut -f1)"
}

do_cs_dialogue() {
  local root="$DATA_DIR/cs_dialogue"
  say "CS-Dialogue (104h 自发对话, BAAI) -> $root  [只取 short_wav]"
  dl BAAI/CS-Dialogue dataset "$root" "data/index/short_wav/*" "data/short_wav/*"
  if [ ! -d "$root/data/short_wav/S0001" ] && ls "$root"/data/short_wav/short_wav.tar.gz* >/dev/null 2>&1; then
    say "解包 short_wav（压缩包顶层是 short_wav/，解到 data/ 才能对上 wav.scp）"
    cat "$root"/data/short_wav/short_wav.tar.gz* | tar xzf - -C "$root/data/"
  else
    say "short_wav 已解包或无分卷，跳过"
  fi
  say "CS-Dialogue 完成：$(du -sh "$root" 2>/dev/null | cut -f1)"
}

do_ascend() {
  local root="$DATA_DIR/ascend"
  say "ASCEND (10.6h 香港自发中英混杂, CAiRE) -> $root"
  dl CAiRE/ASCEND dataset "$root" "main/*.parquet"
  say "ASCEND 完成：$(du -sh "$root" 2>/dev/null | cut -f1)（音频内嵌 parquet，由 prepare_manifests.py 解出）"
}

do_gigaspeech() {
  local root="$DATA_DIR/gigaspeech"
  say "GigaSpeech M (1000h 英文) -> $root"
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "GigaSpeech 是 gated 数据集，需要 HF token。" >&2
    echo "  1) 到 https://huggingface.co/datasets/speechcolab/gigaspeech 同意条款" >&2
    echo "  2) 把 token 放到 /data/.cache/huggingface/token 或设 HF_TOKEN" >&2
    exit 1
  fi
  dl speechcolab/gigaspeech dataset "$root" "parquet-data/m/*.parquet"
  local n; n=$(ls "$root"/parquet-data/m/*.parquet 2>/dev/null | wc -l)
  say "GigaSpeech 完成：$n 个 parquet，$(du -sh "$root" 2>/dev/null | cut -f1)（音频内嵌，由 prepare_manifests.py 解出）"
}

mkdir -p "$DATA_DIR"
case "$TARGET" in
  talcs)        do_talcs ;;
  cs_dialogue)  do_cs_dialogue ;;
  ascend)       do_ascend ;;
  gigaspeech)   do_gigaspeech ;;
  all)          do_talcs; do_cs_dialogue; do_ascend; do_gigaspeech ;;
  *) echo "用法: $0 [talcs|cs_dialogue|ascend|gigaspeech|all]" >&2; exit 1 ;;
esac
say "全部完成。下一步：python prepare_manifests.py --dataset <name> ..."
