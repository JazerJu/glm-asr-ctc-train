"""逐卡体检：H2D 大块拷贝 + 流同步 + 一段 matmul。

起训前跑一遍。多卡训练里坏卡的表现不是「报错」而是「卡住」——
HCCL 会一直等，到 HCCL_EXEC_TIMEOUT（默认 1836s）才拆通信域，
届时崩在别处（torch.load / load_state_dict 的设备拷贝），排查方向全被带偏。
2026-09-08 就为此白烧了两次起训、一个多小时。

    source scripts/_ascend_env.sh
    python scripts/npu_card_check.py            # 查全部 8 张
    python scripts/npu_card_check.py 0 1 2      # 只查指定几张

健康卡约 1.1s 返回。**坏卡是挂住不返回**，不是抛异常 —— 所以带超时，
超时即判定为不可用。挂住的进程会是 D 状态，kill -9 都杀不掉，只能重启整机。
"""
import multiprocessing as mp
import sys
import time

TIMEOUT_SEC = 60.0


def _probe(card, q):
    try:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(card)
        t0 = time.time()
        # 按 ctc_lo.weight（72468x512 fp32，148 MB）和 linear1.weight 的实际形状来，
        # 这两张就是训练里 load_state_dict 拷得最多的
        for shape in [(72468, 512), (2048, 2048)]:
            cpu = torch.randn(*shape, dtype=torch.float32)
            dev = torch.empty(*shape, dtype=torch.float32, device=f"npu:{card}")
            dev.copy_(cpu)
            torch.npu.synchronize()
        a = torch.randn(2048, 2048, device=f"npu:{card}", dtype=torch.float16)
        for _ in range(20):
            a = a @ a.T / 100.0
        torch.npu.synchronize()
        q.put((card, "OK", time.time() - t0, ""))
    except Exception as e:  # noqa: BLE001
        q.put((card, "FAIL", 0.0, f"{type(e).__name__}: {str(e)[:120]}"))


def main():
    cards = [int(x) for x in sys.argv[1:]] or list(range(8))
    bad = []
    for c in cards:
        q = mp.Queue()
        p = mp.Process(target=_probe, args=(c, q))
        p.start()
        p.join(TIMEOUT_SEC)
        if p.is_alive():
            p.terminate(); p.join(5)
            print(f"card {c}  HANG   超过 {TIMEOUT_SEC:.0f}s 未返回 —— 判定不可用")
            bad.append(c)
            continue
        card, status, dt, msg = q.get() if not q.empty() else (c, "FAIL", 0.0, "无返回")
        print(f"card {card}  {status:5s}  {dt:.1f}s  {msg}")
        if status != "OK":
            bad.append(card)

    print()
    if bad:
        ok = [c for c in cards if c not in bad]
        print(f"不可用: {bad}")
        print(f"可用:   {ok}")
        print("起训时排除坏卡（只改 NPROC 无效，torchrun 的 rank N 绑 npu:N）：")
        print(f"    export ASCEND_RT_VISIBLE_DEVICES={','.join(map(str, ok))}")
        print(f"    NPROC_N={len(ok)}")
        sys.exit(1)
    print(f"全部 {len(cards)} 张可用")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
