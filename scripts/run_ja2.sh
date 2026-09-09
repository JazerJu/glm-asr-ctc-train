#!/bin/bash
# 朗读语域补充轮。基座是 v1-ja（checkpoints_v1_ja/best.pt, step 89,463, ffn 128）。
#
# 为什么选 v1-ja 而不是 v2-ja：日语只差 0.43pp（27.69 vs 27.26），但 v1 的中文
# 是四个模型里最好的（5.22%），参数还少 17%（48.3M vs 58.2M）。CapsWriter 是
# 中文听写工具，中文是主场景。见 docs/qwen3-ctc-ja-export-and-inference.md §2。
#
# 这一轮新增三份日语朗读语料，上一轮那 26 条一条不删：
#   cv_ja_other  61,649 条 / 74.6 h  真人朗读，多说话人。other.tsv 里 up>=1 且
#                down==0 的部分 —— 录制流程和 validated 完全相同，没进 validated
#                只是因为 ja 的验证者太少、票数不够，不是另一个领域。
#   jsut          5,000 条 /  6.78 h  真人朗读，单说话人（basic5000）。量小且
#                说话人单一，上采样要克制，不然就是在学一个人的音色。
#   tts_ja       约 4.4 万条 / 约 44 h  FishAudioS2 + FireRedTTS3 合成，维基文本。
#
# 上采样一律 x2。理由：这三份合计约 125 h，相对日语 1,640.7 h 只有 7.1%，
# 不放大基本等于没加；但它们又不是 ReazonSpeech 那种量级的真实语料，x3 以上
# 会开始过表达（尤其 jsut 单说话人和 TTS 的合成痕迹）。x2 后约 250 h，
# 占日语约 12.4%，是个能被看见又不至于主导的比例。
#
# 注意 tts_ja 里同一句会被两个引擎各念一遍 —— 这是有意的增广，不是重复，
# build_tts_ja 按 audio_path 去重而不是按 text。
set -euo pipefail
cd /remote-home/wy008/glm-ctc

BASE="manifests/aishell1.jsonl,manifests/wenetspeech.jsonl,manifests/magicdata.jsonl,manifests/cv_yue.jsonl,manifests/cv_zh_hk.jsonl,manifests/librispeech.jsonl,manifests/ksponspeech.jsonl,manifests/cv_ja.jsonl,manifests/mls_german.jsonl,manifests/mls_dutch.jsonl,manifests/mls_french.jsonl,manifests/mls_spanish.jsonl,manifests/mls_italian.jsonl,manifests/mls_portuguese.jsonl,manifests/mls_polish.jsonl,manifests/cv_zh_tw.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/gigaspeech.jsonl,manifests/reazon_ja_large.jsonl"
READ="manifests/cv_ja_other.jsonl,manifests/cv_ja_other.jsonl,manifests/jsut.jsonl,manifests/jsut.jsonl,manifests/tts_ja.jsonl,manifests/tts_ja.jsonl"
export MANIFESTS="$BASE,$READ"

CKPT=checkpoints_v1_ja/best.pt
RESUME_STEP=89463
FFN_ARG=""                          # v1 系是默认 128，传 2048 会 shape mismatch
export SAVE_DIR=checkpoints_v1_ja_read
RUN_NAME=qwen-asr-ctc-v1-ja-read

# 2026-09-08：card 5 的 npu-smi Health 变成 Alarm，逐卡测 H2D 拷贝时它是唯一
# 挂住不返回的（其余七张各 1.1s），进程卡在不可中断的设备调用里 kill -9 都杀不掉。
# 两次起训失败都是 rank5 第一个报 ACL stream synchronize failed (507034/507048)，
# 对得上。不动硬件，直接把它从可见设备里挖掉走 7 卡。
# torchrun 只认 nproc_per_node，rank N 绑 npu:N —— 只把 NPROC 改成 7 会用 0..6，
# 照样踩到坏卡，必须靠 ASCEND_RT_VISIBLE_DEVICES 做重映射。
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,6,7
BATCH_PER_CARD=32; NPROC_N=7
TOTAL=$(python3 -c "
import os
print(sum(sum(1 for _ in open(m)) for m in os.environ['MANIFESTS'].split(',') if m))")
PER_STEP=$(( BATCH_PER_CARD * NPROC_N ))
STEPS=$(( TOTAL / PER_STEP ))
END=$(( RESUME_STEP + STEPS ))

# 峰值 LR 从「目标起始 LR」反算。调度器是 0.5*(1+cos(pi*step/max_steps))，
# 分母不减 warmup，接续时 step 已经很大，同一个峰值在不同 resume 点差好几倍。
# 这一轮起始 LR 压到 6e-5（上一轮是 1.1e-4）：v1-ja 已经在日语上收敛得不错，
# 新加的只有 250 h、占比 1.6%，用大 LR 会把已经学到的东西抖散。
TARGET_START_LR=6e-5
PEAK_LR=$(python3 -c "
import math
f = 0.5 * (1 + math.cos(math.pi * $RESUME_STEP / $END))
print('%.6e' % ($TARGET_START_LR / f))")

echo "基座 $CKPT  样本 $TOTAL  本轮 $STEPS 步  $RESUME_STEP -> $END"
echo "峰值 LR $PEAK_LR  ->  起始 LR 约 $TARGET_START_LR"
echo "W&B run: $RUN_NAME"

# 2026-09-08 首次起训在这里炸过：8 rank 同时把 checkpoint 搬进 NPU，
# HCCL 默认 1836s 的执行超时到点拆通信域。已把 torch.load 改成读到 CPU，
# 这条是保险 —— 启动阶段（加载 33 个 manifest、870 万样本）本来就慢。
export HCCL_EXEC_TIMEOUT=3600
export NPROC=$NPROC_N BATCH=$BATCH_PER_CARD GRAD_ACCUM=1
export EPOCHS=1 WARMUP_EPOCHS=0
export SAVE_INTERVAL=2000 KEEP_LAST=5
export WANDB_PROJECT=qwen3-asr-ctc
export EXTRA_ARGS="$FFN_ARG --resume $CKPT --resume-lr $PEAK_LR --lr-max-steps $END --wandb-log-checkpoints --wandb-checkpoint-every 4000 --wandb-run-name $RUN_NAME"
exec bash scripts/run_ddp_ascend.sh
