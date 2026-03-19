import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# ========= 你只改这里 =========
INPUT_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\dca_need\stacking_ablation_6bio_vs_full_pred_test.csv"
OUTPUT_CSV = r"D:\yiqy\sleepProjexts\发文章版_疼痛二分类任务\最终版本\outputs\dca_need\toy_6bio_15bio_probs.csv"

TARGETS = {
    "p_ensemble_6bio": {"auc": 0.729, "ap": 0.557},
    "p_ensemble_full": {"auc": 0.819, "ap": 0.826},
}

SEED = 42
N_RANDOM_SEARCH = 6000
# ============================


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def score_loss(y, p, target_auc, target_ap):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    # AUC 权重稍高一点
    loss = 4.0 * (auc - target_auc) ** 2 + 1.0 * (ap - target_ap) ** 2
    return loss, auc, ap


def make_candidate_probs(y, rng, mu_pos, mu_neg, sd_pos, sd_neg, sharpness=1.0, shift=0.0):
    z = np.empty(len(y), dtype=float)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    z[pos_idx] = rng.normal(mu_pos, sd_pos, size=len(pos_idx))
    z[neg_idx] = rng.normal(mu_neg, sd_neg, size=len(neg_idx))

    # 控制概率“拉伸/压缩”
    z = sharpness * z + shift
    p = sigmoid(z)

    # 裁剪到合理范围，避免 0/1 极端值太多
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return p


def search_probs(y, target_auc, target_ap, seed=42, n_iter=6000):
    rng = np.random.default_rng(seed)
    best = None

    for i in range(n_iter):
        # 随机采样参数
        mu_neg = rng.uniform(-2.0, 0.5)
        gap = rng.uniform(0.4, 3.5)
        mu_pos = mu_neg + gap

        sd_neg = rng.uniform(0.3, 1.5)
        sd_pos = rng.uniform(0.3, 1.5)

        sharpness = rng.uniform(0.7, 1.8)
        shift = rng.uniform(-0.8, 0.8)

        p = make_candidate_probs(
            y=y,
            rng=rng,
            mu_pos=mu_pos,
            mu_neg=mu_neg,
            sd_pos=sd_pos,
            sd_neg=sd_neg,
            sharpness=sharpness,
            shift=shift
        )

        loss, auc, ap = score_loss(y, p, target_auc, target_ap)

        # 附加一点轻微约束，避免太离谱
        brier = brier_score_loss(y, p)
        loss += 0.05 * (brier - 0.20) ** 2

        if best is None or loss < best["loss"]:
            best = {
                "loss": loss,
                "auc": auc,
                "ap": ap,
                "brier": brier,
                "p": p.copy(),
                "params": {
                    "mu_neg": mu_neg,
                    "mu_pos": mu_pos,
                    "sd_neg": sd_neg,
                    "sd_pos": sd_pos,
                    "sharpness": sharpness,
                    "shift": shift,
                }
            }

    return best


def main():
    df = pd.read_csv(INPUT_CSV)

    if "y_true" not in df.columns:
        raise ValueError("输入文件必须包含 y_true 列")

    y = df["y_true"].astype(int).values

    # 先生成 6bio
    best_6 = search_probs(
        y=y,
        target_auc=TARGETS["p_ensemble_6bio"]["auc"],
        target_ap=TARGETS["p_ensemble_6bio"]["ap"],
        seed=SEED,
        n_iter=N_RANDOM_SEARCH
    )

    # 再生成 15bio
    best_15 = search_probs(
        y=y,
        target_auc=TARGETS["p_ensemble_full"]["auc"],
        target_ap=TARGETS["p_ensemble_full"]["ap"],
        seed=SEED + 1,
        n_iter=N_RANDOM_SEARCH
    )

    out = pd.DataFrame({
        "y_true": y,
        "p_ensemble_full": best_15["p"],
        "p_ensemble_6bio": best_6["p"],
    })

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", float_format="%.6f")

    print("=" * 80)
    print("Toy file saved to:")
    print(OUTPUT_CSV)
    print("-" * 80)
    print("6 biomarkers target/result")
    print(f"Target AUC: {TARGETS['p_ensemble_6bio']['auc']:.3f},  Result AUC: {best_6['auc']:.3f}")
    print(f"Target AP : {TARGETS['p_ensemble_6bio']['ap']:.3f},   Result AP : {best_6['ap']:.3f}")
    print(f"Result Brier: {best_6['brier']:.3f}")
    print("-" * 80)
    print("15 biomarkers target/result")
    print(f"Target AUC: {TARGETS['p_ensemble_full']['auc']:.3f},  Result AUC: {best_15['auc']:.3f}")
    print(f"Target AP : {TARGETS['p_ensemble_full']['ap']:.3f},   Result AP : {best_15['ap']:.3f}")
    print(f"Result Brier: {best_15['brier']:.3f}")
    print("=" * 80)
    print("仅用于调试画图函数，不可替代真实研究结果。")


if __name__ == "__main__":
    main()