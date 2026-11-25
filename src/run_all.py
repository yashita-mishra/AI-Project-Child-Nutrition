"""Run every experiment from the repo root:   python -m src.run_all
Writes results/*.csv, results/model.json and figures/*.png. Takes about two minutes."""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import experiments as E
from .data import NODES, STATES, PARENTS, ROOT, load, make_net

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
COL = {"em": "#3d7a4a", "deletion": "#c67139", "mode": "#8a8a8a", "bn_em": "#3d7a4a", "bn_deletion": "#c67139",
       "logistic": "#5b7fa6", "prevalence": "#8a8a8a"}
LAB = {"em": "EM (keep gaps)", "deletion": "Delete incomplete rows", "mode": "Fill with most common value",
       "bn_em": "Network + EM", "bn_deletion": "Network, complete cases", "logistic": "Logistic regression",
       "prevalence": "Prevalence only"}


def main(seeds=30, boot=200, ui_boot=100):
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    t0 = time.time()
    d = load()
    X = make_net().encode(d)
    log = {}

    # ---- 0. prior strength by cross-validation
    grid = E.choose_strength(d)
    grid.to_csv(f"{RES}/prior_strength_cv.csv", index=False)
    strength = float(grid.sort_values("logloss").iloc[0].strength)
    log["prior_strength"] = strength
    print("prior strength grid:\n", grid.round(4).to_string(index=False), "-> chosen", strength)

    # ---- 1. final model
    net = make_net().fit_em(X, strength=strength)
    json.dump(net.to_dict(), open(f"{RES}/model_cpts.json", "w"))
    rows = []
    for n in NODES:
        for c in range(net.q[n]):
            cfg = " & ".join(f"{p}={STATES[p][i]}" for p, i in zip(PARENTS[n], np.unravel_index(c, [len(STATES[p]) for p in PARENTS[n]]))) if PARENTS[n] else "(none)"
            for s, st in enumerate(STATES[n]):
                rows.append(dict(node=n, parent_config=cfg, state=st, P_EM=round(float(net.cpt[n][c, s]), 5)))
    pd.DataFrame(rows).to_csv(f"{RES}/cpt_em_final.csv", index=False)

    # ---- 2. masking experiments
    rec = E.recovery_experiment(d, seeds=seeds, strength=strength)
    rec.to_csv(f"{RES}/masking_recovery_raw.csv", index=False)
    rec_s = E.summarise_recovery(rec); rec_s.to_csv(f"{RES}/masking_recovery.csv", index=False)
    down, ref_fx = E.downstream_experiment(d, seeds=seeds, strength=strength)
    down.to_csv(f"{RES}/masking_downstream_raw.csv", index=False)
    down_s = E.summarise_downstream(down); down_s.to_csv(f"{RES}/masking_downstream.csv", index=False)
    log["reference_food_security_effect"] = ref_fx
    print("masking done", round(time.time() - t0, 1), "s")

    # ---- 3. held-out validation
    preds = E.cross_validate(d, strength=strength, repeats=20)
    preds.to_csv(f"{RES}/heldout_predictions.csv", index=False)
    sc = E.score(preds); sc.to_csv(f"{RES}/heldout_scores.csv", index=False)
    cal, ece = E.calibration(preds); cal.to_csv(f"{RES}/heldout_calibration.csv", index=False)
    log["calibration_ece"] = ece
    print(sc.round(3).to_string(index=False), "\nECE", round(ece, 4))

    # ---- 4. model checks
    chk = E.ci_checks(d, net); chk.to_csv(f"{RES}/implied_independence_checks.csv", index=False)

    # ---- 5. scenarios with bootstrap intervals
    base, boots = E.bootstrap_nets(d, reps=max(boot, ui_boot), strength=strength)
    scn = E.run_scenarios(base, boots[:boot]); scn.to_csv(f"{RES}/scenarios.csv", index=False)

    # ---- 6. UI payload
    ui = dict(model=net.to_dict(), boot=[{n: np.round(b.cpt[n], 5).tolist() for n in NODES} for b in boots[:ui_boot]],
              scenarios=scn.round(5).to_dict("records"),
              validation=dict(scores=sc.round(4).to_dict("records"), calibration=cal.round(4).to_dict("records"), ece=round(ece, 4),
                              n_labelled=int((X[:, -1] >= 0).sum()), n_positive=int((X[:, -1] == 1).sum()), n=len(X)),
              masking=dict(recovery=rec_s.round(4).to_dict("records"), downstream=down_s.round(4).to_dict("records"),
                           reference_fs_effect=round(ref_fx, 5)),
              prior_strength=strength)
    json.dump(ui, open(f"{RES}/ui_payload.json", "w"))
    json.dump(log, open(f"{RES}/run_log.json", "w"), indent=1)

    figures(rec_s, down_s, sc, cal, ece, scn, preds)
    print("all done in", round(time.time() - t0, 1), "s")


# ======================================================================== figures
def figures(rec_s, down_s, sc, cal, ece, scn, preds):
    pats = [("mcar", "Random (MCAR)"), ("mar", "Depends on ethnicity (MAR)"), ("mnar", "Depends on the value (MNAR)")]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for k, (p, title) in enumerate(pats):
        for m in ("em", "deletion", "mode"):
            g = rec_s[(rec_s.pattern == p) & (rec_s.method == m)].sort_values("rate")
            ax[0, k].errorbar(g.rate * 100, g.accuracy, yerr=g.accuracy_sd, marker="o", color=COL[m], label=LAB[m], capsize=3)
            ax[1, k].errorbar(g.rate * 100, g.logloss, yerr=g.logloss_sd, marker="o", color=COL[m], capsize=3)
        ax[0, k].set_title(title); ax[1, k].set_xlabel("% of covariate values hidden")
        ax[0, k].grid(alpha=.3); ax[1, k].grid(alpha=.3)
    ax[0, 0].set_ylabel("Accuracy recovering hidden values"); ax[1, 0].set_ylabel("Log loss (lower is better)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Masking experiment: recovering hidden covariate values (mean ± sd over 30 random masks)")
    plt.tight_layout(); plt.savefig(f"{FIG}/masking_recovery.png", dpi=130); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for k, (p, title) in enumerate(pats):
        for m in ("em", "deletion", "mode"):
            g = down_s[(down_s.pattern == p) & (down_s.method == m)].sort_values("rate")
            ax[0, k].errorbar(g.rate * 100, g.risk_table_mae * 100, yerr=g.risk_table_mae_sd * 100, marker="o", color=COL[m], capsize=3, label=LAB[m])
            ax[1, k].errorbar(g.rate * 100, g.fs_effect_mean * 100, yerr=g.fs_effect_sd * 100, marker="o", color=COL[m], capsize=3)
        ax[1, k].axhline(100 * down_s.fs_effect_mean[(down_s.method == "em") & (down_s.rate == down_s.rate.min())].mean() * 0 + 100 * json.load(open(f"{RES}/run_log.json"))["reference_food_security_effect"],
                         color="k", ls=":", lw=1)
        ax[0, k].set_title(title); ax[1, k].set_xlabel("% of covariate values hidden"); ax[0, k].grid(alpha=.3); ax[1, k].grid(alpha=.3)
    ax[0, 0].set_ylabel("Error in P(anaemia | parents)\n(percentage points)"); ax[1, 0].set_ylabel("Food-security effect on anaemia\n(percentage points; dotted = unmasked data)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Masking experiment: does the way gaps are handled change the conclusions?")
    plt.tight_layout(); plt.savefig(f"{FIG}/masking_downstream.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].plot([0, 0.5], [0, 0.5], "k:", lw=1)
    ax[0].plot(cal.mean_predicted, cal.observed_rate, "o-", color=COL["bn_em"])
    for _, r in cal.iterrows():
        ax[0].annotate(f"n={int(r.n)}, {int(r.positives)}+", (r.mean_predicted, r.observed_rate), fontsize=7, xytext=(4, -10), textcoords="offset points")
    ax[0].set_xlabel("Predicted risk"); ax[0].set_ylabel("Observed anaemia rate"); ax[0].set_title(f"Calibration on held-out children (ECE {ece:.3f})"); ax[0].grid(alpha=.3)
    ms = ["bn_em", "bn_deletion", "logistic", "prevalence"]
    ax[1].bar([LAB[m] for m in ms], sc.set_index("method").loc[ms].auc, yerr=sc.set_index("method").loc[ms].auc_sd, color=[COL[m] for m in ms], capsize=3)
    ax[1].axhline(.5, color="k", ls=":"); ax[1].set_ylim(.4, .9); ax[1].set_title("Held-out AUC (mean ± sd over repeats)"); ax[1].tick_params(axis="x", labelrotation=25, labelsize=8)
    ax[2].bar([LAB[m] for m in ms], sc.set_index("method").loc[ms].logloss, yerr=sc.set_index("method").loc[ms].logloss_sd, color=[COL[m] for m in ms], capsize=3)
    ax[2].set_title("Held-out log loss (lower is better)"); ax[2].tick_params(axis="x", labelrotation=25, labelsize=8)
    ax[2].set_ylim(sc.logloss.min() - .05, sc.logloss.max() + .05)
    plt.tight_layout(); plt.savefig(f"{FIG}/heldout_validation.png", dpi=130); plt.close()

    s = scn[~scn.arm.str.startswith("Change") & ~scn.scenario.eq("revision")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r.title[:44]}\n{r.arm}" for r in s.itertuples()]
    y = np.arange(len(s))[::-1]
    ax.errorbar(s.risk * 100, y, xerr=[(s.risk - s.lo) * 100, (s.hi - s.risk) * 100], fmt="o", color="#3d7a4a", capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7); ax.set_xlabel("Predicted anaemia risk (%), 95% bootstrap interval"); ax.grid(alpha=.3)
    ax.set_title("What-if scenarios")
    plt.tight_layout(); plt.savefig(f"{FIG}/scenarios.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
