#!/bin/bash
# Self-conditioned CTC 轮。基座是 v1-ja-read2（step 176,491, val_loss 0.5455）。
#
# 数据一个字没动，和上一轮完全相同的 37 个 manifest。这一轮只动训练方法，
# 因为数据这条路已经走到边际收益很低的地方了：
#     +1,159.6 h 日语 → ReazonSpeech CER 只降 1.67pp（每小时 0.01038 → 0.00141）
# 而完整 Qwen3-ASR 用同一个冻结编码器能做到 5.35%，说明瓶颈不在编码器表征，
# 在 CTC 头怎么用它。
#
# 四处改动：
#
# 1) --self-cond + --inter-ctc-weight 0.3   （arXiv 2104.02724，Self-conditioned CTC）
#    这两个开关是一件事的两半，必须一起开：中间层（下标 1、3）的预测既拿去算
#    辅助 CTC 损失（权重 0.3），又经 conditioning_layer 投影回去加到下一层输入。
#    ESPnet 里对应 interctc_weight + interctc_use_conditioning=True，
#    共用同一组 interctc_layer_idx。只开 --self-cond 而权重为 0 是错的配置——
#    中间预测没有监督信号，回流的是没训过的东西。
#    conditioning_layer 是独立的 nn.Linear(72468, 512)，所有注入点共享一个，
#    照官方实现来，不是权重绑定版（那个文献没验证过）。
#    代价：参数 48.3M → 85.4M；GPU 推理 +3~7%（ctc_lo 在 GPU 上只占头部 8.8%）。
#    论文消融（贪心解码、无 LM）：
#        TEDLIUM2  12.2 → 10.1(InterCTC) → 9.4(Self-cond)
#        WSJ       14.9 → 12.7           → 11.9
#        AISHELL-1  6.2 →  5.7           →  5.3
#
# 2) --epochs 2
#    前六轮每轮都恰好 1 个 epoch，从没试过第二遍。上一轮 epoch 末尾 train loss
#    0.5724 仍在以每 5k 步 0.002 的速度下降，且高于 val loss 0.5455 —— 是欠拟合，
#    不是过拟合。数据没变的情况下，第二遍是最便宜的一项。
#
# 3) --specaug
#    论文那三组基线本身就带 SpecAugment，上面的增益是叠加在它之上取得的。
#    F=16 / time_ratio=0.1 是按我们 128 维 mel 重新标定过的（论文的 F=27 在
#    128 维上遮蔽率 37.4%，太狠；现在中位 18.6%）。
#
# 4) --ja-val-manifest
#    主 val 是从训练集随机切的，混了 15 个语种，日语只占 18.5%，跨轮还因为
#    数据集变化不可比。加一条日语专用监控（reazon_ja_test，5,263 条抽 2,000），
#    每个 epoch 记一次 epoch/ja_val_loss。**只做监控，不参与 best.pt 选择**——
#    它是我们对外报 CER 的那个测试集，不能拿来选模型。
set -euo pipefail
cd /remote-home/wy008/glm-ctc

# 2026-09-09 二次体检：card 6 自己好了（前一天挂的），现在只有 card 5 坏。
# 卡的状态天天在变，别照抄下面这行，起训前自己跑：
#     bash -c "source scripts/_ascend_env.sh; python scripts/npu_card_check.py"
# 注意不 source 环境直接跑会八张卡全报 torch_npu 加载失败，那是假阳性。
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
export SAVE_DIR=checkpoints_v1_ja_sc
RUN_NAME=qwen-asr-ctc-v1-ja-sc

TOTAL=$(python3 -c "
import os
print(sum(sum(1 for _ in open(m)) for m in os.environ['MANIFESTS'].split(',') if m))")
PER_STEP=$(( BATCH_PER_CARD * NPROC_N ))
STEPS=$(( TOTAL * N_EPOCHS / PER_STEP ))
END=$(( RESUME_STEP + STEPS ))

# 峰值 LR 从目标起始 LR 反算，调度器 0.5*(1+cos(pi*step/max_steps))。
# 仍用 6e-5：这一轮变的是方法，LR 跟着动会没法归因。
TARGET_START_LR=6e-5
PEAK_LR=$(python3 -c "
import math
f = 0.5 * (1 + math.cos(math.pi * $RESUME_STEP / $END))
print('%.6e' % ($TARGET_START_LR / f))")

echo "基座 $CKPT  样本 $TOTAL  x$N_EPOCHS epoch = $STEPS 步  $RESUME_STEP -> $END"
echo "卡 $ASCEND_RT_VISIBLE_DEVICES  ($NPROC_N 张)"
echo "峰值 LR $PEAK_LR  ->  起始 LR 约 $TARGET_START_LR"
echo "W&B run: $RUN_NAME"

export NPROC=$NPROC_N BATCH=$BATCH_PER_CARD GRAD_ACCUM=1
export EPOCHS=$N_EPOCHS WARMUP_EPOCHS=0
export SAVE_INTERVAL=2000 KEEP_LAST=5
export WANDB_PROJECT=qwen3-asr-ctc
export EXTRA_ARGS="--resume $CKPT --resume-lr $PEAK_LR --lr-max-steps $END \
--self-cond --inter-ctc-weight 0.3 --specaug \
--ja-val-manifest manifests_test/reazon_ja_test.jsonl --ja-val-max 2000 \
--wandb-log-checkpoints --wandb-checkpoint-every 4000 --wandb-run-name $RUN_NAME"
exec bash scripts/run_ddp_ascend.sh
