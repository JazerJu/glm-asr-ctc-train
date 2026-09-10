#!/bin/bash
# Self-cond 消融轮。除了不开 --self-cond，其余与 run_ja4.sh 逐项相同。
#
# 目的：把 Self-conditioned CTC 单独的贡献剥出来。上一轮同时动了三样
# （自条件、第 2 个 epoch、SpecAugment），涨了但归因不清。
#
# 只消融自条件，不消融另外两样，是因为**只有它有推理代价**：
#   InterCTC     训练期辅助损失，推理图不变，零开销
#   SpecAugment  训练期数据增广，推理图不变，零开销
#   Self-cond    conditioning_layer 参与前向，+37.1M 参数、int4 +19 MB、GPU +3~7%
# 前两样就算贡献是 0 也不用为它们付账；这一项要付，所以必须证明值得。
#
# ── 可比性：下面这些一个都不能动 ──────────────────────────────────
#
# 基座必须是 read2，不是 ja-sc。消融的定义是两条支路共享同一个起点：
#     read2 best.pt ──┬── +自条件 ──> v1-ja-sc    （已完成，FLEURS ja 19.2）
#                     └── −自条件 ──> 本轮
# 从 ja-sc 接续测的是「把训好的回流拆掉」，是另一个问题；而且它的 state_dict
# 里带 conditioning_layer.*，不开 --self-cond 会当成 unexpected 直接报错。
#
# 卡数必须是 7 张。2026-09-10 体检时 8 张全好，但用 8 张会把有效 batch 从
# 224 变成 256，测出来的差就不只是自条件了。设备号也照抄上一轮的 0-4,6,7。
#
# END 和 PEAK_LR 直接写死上一轮算出来的值，不重算。原脚本是按 TOTAL*2/PER_STEP
# 反算的，卡数或 manifest 行数有任何变化都会让 LR 曲线整条偏掉。
set -euo pipefail
cd /remote-home/wy008/glm-ctc

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,6,7
BATCH_PER_CARD=32; NPROC_N=7

export HCCL_EXEC_TIMEOUT=3600

BASE="manifests/aishell1.jsonl,manifests/wenetspeech.jsonl,manifests/magicdata.jsonl,manifests/cv_yue.jsonl,manifests/cv_zh_hk.jsonl,manifests/librispeech.jsonl,manifests/ksponspeech.jsonl,manifests/cv_ja.jsonl,manifests/mls_german.jsonl,manifests/mls_dutch.jsonl,manifests/mls_french.jsonl,manifests/mls_spanish.jsonl,manifests/mls_italian.jsonl,manifests/mls_portuguese.jsonl,manifests/mls_polish.jsonl,manifests/cv_zh_tw.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/talcs.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/cs_dialogue.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/ascend.jsonl,manifests/gigaspeech.jsonl,manifests/reazon_ja_large.jsonl"
NEWBC="manifests/reazon_ja_all2.jsonl"
READ="manifests/cv_ja_other.jsonl,manifests/cv_ja_other.jsonl,manifests/cv_ja_other.jsonl,manifests/jsut.jsonl,manifests/jsut.jsonl,manifests/jsut.jsonl,manifests/tts_ja.jsonl,manifests/tts_ja.jsonl,manifests/tts_ja.jsonl"
export MANIFESTS="$BASE,$NEWBC,$READ"

CKPT=checkpoints_v1_ja_read2/best.pt
RESUME_STEP=176491
N_EPOCHS=2
END=262451                          # 与 run_ja4.sh 实跑值一致，不重算
PEAK_LR=2.477860e-04                # 同上；对应起始 LR 6e-5
export SAVE_DIR=checkpoints_v1_ja_nosc
RUN_NAME=qwen-asr-ctc-v1-ja-nosc

echo "基座 $CKPT   $RESUME_STEP -> $END   x$N_EPOCHS epoch"
echo "卡 $ASCEND_RT_VISIBLE_DEVICES  ($NPROC_N 张，故意不用满 8 张)"
echo "峰值 LR $PEAK_LR  ->  起始 LR 约 6e-5"
echo "W&B run: $RUN_NAME"
echo "本轮 **不开** --self-cond，其余与 run_ja4.sh 相同"

export NPROC=$NPROC_N BATCH=$BATCH_PER_CARD GRAD_ACCUM=1
export EPOCHS=$N_EPOCHS WARMUP_EPOCHS=0
export SAVE_INTERVAL=2000 KEEP_LAST=5
export WANDB_PROJECT=qwen3-asr-ctc
export EXTRA_ARGS="--resume $CKPT --resume-lr $PEAK_LR --lr-max-steps $END \
--inter-ctc-weight 0.3 --specaug \
--ja-val-manifest manifests_test/reazon_ja_test.jsonl --ja-val-max 2000 \
--wandb-log-checkpoints --wandb-checkpoint-every 4000 --wandb-run-name $RUN_NAME"
exec bash scripts/run_ddp_ascend.sh
