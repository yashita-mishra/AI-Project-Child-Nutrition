"""Masking experiment, held-out validation, scenarios with bootstrap intervals."""
import numpy as np
import pandas as pd
from scipy import stats

from .data import COVARIATES, NODES, STATES, load, make_net

STRENGTH = 2.0      # Dirichlet prior strength per parent configuration (chosen by cross-validation, see run_all)


# ============================================================================ helpers
def fit_em(X, strength=STRENGTH, nodes=NODES, init=None):
    net = make_net(nodes)
    cols = [NODES.index(n) for n in nodes]
    return net.fit_em(X[:, cols], strength=strength, init=init)


def risk_table(net):
    """P(anaemia=yes | ethnicity, food_security, wic_child) for all 20 configurations of its parents."""
    return net.cpt["anaemia"][:, 1]


def fs_effect(net):
    """Causal contrast P(anaemia | do(insecure)) - P(anaemia | do(secure))."""
    return net.ve_query("anaemia", None, {"food_security": "insecure"})[1] - \
        net.ve_query("anaemia", None, {"food_security": "secure"})[1]


# ============================================================================ masking patterns
def mask_cells(X, pattern, rate, rng, nodes=COVARIATES):
    """Hide a share of the observed covariate cells. Returns a masked copy and the boolean mask.
    mcar : every cell equally likely.
    mar  : likelihood depends on an always-observed variable (ethnicity): Black and Other-Hispanic
           children are hidden 3x as often.
    mnar : likelihood depends on the value itself: the 'worse' value (insecure, below poverty, ...)
           is hidden 3x as often. EM is not designed for this; it is a stress test."""
    Xm = X.copy()
    mask = np.zeros(X.shape, dtype=bool)
    eth = X[:, NODES.index("ethnicity")]
    for n in nodes:
        j = NODES.index(n)
        obs = X[:, j] >= 0
        w = np.ones(len(X))
        if pattern == "mar":
            w = np.where(np.isin(eth, [1, 3]), 3.0, 1.0)          # OthHisp=1, NHBlack=3
        elif pattern == "mnar":
            bad = {"income": 0, "food_security": 1, "wic_child": 0, "snap_past_year": 0, "parent_edu": 0, "ethnicity": 3}[n]
            w = np.where(X[:, j] == bad, 3.0, 1.0)
        p = np.clip(rate * w / w[obs].mean(), 0, 0.95)
        hit = obs & (rng.random(len(X)) < p)
        mask[hit, j] = True
    Xm[mask] = -1
    return Xm, mask


# ============================================================================ experiment 1: recover masked cells
def recovery_experiment(d, rates=(0.1, 0.3, 0.5), patterns=("mcar", "mar", "mnar"), seeds=30, strength=STRENGTH):
    """How well is a hidden covariate value recovered?  Network over the 6 covariates (anaemia left out so
    the comparison is about the covariates alone).
      em        : EM fit on all rows, gaps kept; prediction = posterior given the rest of the row.
      deletion  : CPTs counted on complete rows only (listwise deletion); same posterior.
      mode      : simple imputation, the most common value of the column (marginal distribution as probabilities)."""
    full = make_net(COVARIATES).encode(d)                      # covariates only
    X = np.full((len(d), len(NODES)), -1, dtype=np.int16)
    X[:, :len(COVARIATES)] = full
    rows = []
    for pattern in patterns:
        for rate in rates:
            for seed in range(seeds):
                rng = np.random.default_rng(1000 * seed + int(rate * 100))
                Xm, mask = mask_cells(X, pattern, rate, rng)
                Xc = Xm[:, :len(COVARIATES)]
                models = {}
                models["em"] = make_net(COVARIATES).fit_em(Xc, strength=strength)
                complete = Xc[(Xc >= 0).all(axis=1)]
                n_complete = len(complete)
                models["deletion"] = make_net(COVARIATES).fit_counts(complete, strength=strength)
                for j, n in enumerate(COVARIATES):
                    cells = np.where(mask[:, j])[0]
                    if len(cells) == 0:
                        continue
                    marg = np.bincount(Xc[Xc[:, j] >= 0, j], minlength=len(STATES[n])).astype(float)
                    marg = (marg + 0.5) / (marg + 0.5).sum()
                    truth = X[cells, j]
                    preds = {"mode": np.tile(marg, (len(cells), 1))}
                    for m in ("em", "deletion"):
                        net = models[m]
                        P = np.zeros((len(cells), len(STATES[n])))
                        for i, c in enumerate(cells):
                            ev = {COVARIATES[k]: int(Xc[c, k]) for k in range(len(COVARIATES)) if Xc[c, k] >= 0 and k != j}
                            P[i] = net.ve_query(n, ev)
                        preds[m] = P
                    for m, P in preds.items():
                        rows.append(dict(pattern=pattern, rate=rate, seed=seed, method=m, node=n, cells=len(cells),
                                         complete_rows=n_complete,
                                         correct=int((P.argmax(1) == truth).sum()),
                                         logloss=float(-np.log(P[np.arange(len(cells)), truth]).sum())))
    return pd.DataFrame(rows)


def summarise_recovery(df):
    g = df.groupby(["pattern", "rate", "method", "seed"]).agg(cells=("cells", "sum"), correct=("correct", "sum"),
                                                              logloss=("logloss", "sum"),
                                                              complete_rows=("complete_rows", "first")).reset_index()
    g["accuracy"] = g.correct / g.cells
    g["logloss"] = g.logloss / g.cells
    out = g.groupby(["pattern", "rate", "method"]).agg(
        accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        logloss=("logloss", "mean"), logloss_sd=("logloss", "std"),
        complete_rows=("complete_rows", "mean")).reset_index()
    return out


# ============================================================================ experiment 2: does the method change the conclusions?
def downstream_experiment(d, rates=(0.1, 0.3, 0.5), patterns=("mcar", "mar", "mnar"), seeds=30, strength=STRENGTH):
    """Mask covariates, keep anaemia, refit the full network three ways and compare with the fit on
    the unmasked data.
      em        : EM on all rows.
      deletion  : complete cases only (every node observed, anaemia included).
      mode      : fill every empty covariate with its most common value, then count (anaemia stays as observed).
    Error measures: (a) mean |difference| in P(anaemia | parents) over the 20 parent configurations, weighted by
    how common each configuration is; (b) error in the food-security causal contrast."""
    X = make_net().encode(d)
    ref = make_net().fit_em(X, strength=strength)
    ref_tab, ref_fx = risk_table(ref), fs_effect(ref)
    cfg = ref._cfg["anaemia"]
    w_cfg = np.bincount(cfg, weights=np.exp(ref._log_joint()), minlength=ref.q["anaemia"])
    w_cfg = w_cfg / w_cfg.sum()
    rows = []
    for pattern in patterns:
        for rate in rates:
            for seed in range(seeds):
                rng = np.random.default_rng(7000 + 1000 * seed + int(rate * 100))
                Xm, _ = mask_cells(X, pattern, rate, rng)
                nets = {"em": make_net().fit_em(Xm, strength=strength)}
                comp = Xm[(Xm >= 0).all(axis=1)]
                nets["deletion"] = make_net().fit_counts(comp, strength=strength)
                Xi = Xm.copy()
                for j in range(len(COVARIATES)):
                    obs = Xi[:, j] >= 0
                    Xi[~obs, j] = np.bincount(Xi[obs, j]).argmax()
                nets["mode"] = make_net().fit_counts(Xi, strength=strength)
                for m, net in nets.items():
                    tab, fx = risk_table(net), fs_effect(net)
                    rows.append(dict(pattern=pattern, rate=rate, seed=seed, method=m,
                                     complete_rows=len(comp),
                                     risk_table_mae=float((w_cfg * np.abs(tab - ref_tab)).sum()),
                                     fs_effect=float(fx), fs_effect_error=float(fx - ref_fx)))
    out = pd.DataFrame(rows)
    out.attrs["reference_fs_effect"] = float(ref_fx)
    return out, float(ref_fx)


def summarise_downstream(df):
    return df.groupby(["pattern", "rate", "method"]).agg(
        risk_table_mae=("risk_table_mae", "mean"), risk_table_mae_sd=("risk_table_mae", "std"),
        fs_effect_mean=("fs_effect", "mean"), fs_effect_sd=("fs_effect", "std"),
        fs_effect_abs_error=("fs_effect_error", lambda s: float(np.abs(s).mean())),
        complete_rows=("complete_rows", "mean")).reset_index()


# ============================================================================ experiment 3: held-out validation
def _auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def cross_validate(d, strength=STRENGTH, repeats=10, folds=5, seed=0):
    """Repeated stratified k-fold on the children with a measured hemoglobin. Held-out children keep their
    covariates as evidence but their anaemia status is hidden and never used for fitting; children without a blood
    test are always in the training data (EM uses their covariates)."""
    X = make_net().encode(d)
    j = NODES.index("anaemia")
    labelled = np.where(X[:, j] >= 0)[0]
    unl = np.where(X[:, j] < 0)[0]
    y_all = X[:, j]
    preds = []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        pos = rng.permutation(labelled[y_all[labelled] == 1])
        neg = rng.permutation(labelled[y_all[labelled] == 0])
        fold_of = {}
        for k, i in enumerate(pos):
            fold_of[i] = k % folds
        for k, i in enumerate(neg):
            fold_of[i] = k % folds
        for f in range(folds):
            test = np.array([i for i in labelled if fold_of[i] == f])
            train = np.setdiff1d(np.arange(len(X)), test)
            em = make_net().fit_em(X[train], strength=strength)
            comp = X[train][(X[train] >= 0).all(axis=1)]
            cc = make_net().fit_counts(comp, strength=strength)
            prev = y_all[np.intersect1d(train, labelled)].mean()
            # logistic regression on one-hot covariates with mode imputation, as a standard baseline
            from sklearn.linear_model import LogisticRegression
            Xtr, Xte = _onehot(X[train]), _onehot(X[test], fit_on=X[train])
            tr_lab = X[train][:, j] >= 0
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr[tr_lab], X[train][tr_lab, j])
            p_lr = lr.predict_proba(Xte)[:, 1]
            for k, i in enumerate(test):
                ev = {NODES[c]: int(X[i, c]) for c in range(len(NODES)) if c != j and X[i, c] >= 0}
                preds.append(dict(repeat=r, fold=f, id=i, y=int(y_all[i]),
                                  bn_em=float(em.ve_query("anaemia", ev)[1]),
                                  bn_deletion=float(cc.ve_query("anaemia", ev)[1]),
                                  logistic=float(p_lr[k]), prevalence=float(prev)))
    return pd.DataFrame(preds)


def _onehot(X, fit_on=None):
    ref = X if fit_on is None else fit_on
    cols = []
    for j in range(len(COVARIATES)):
        obs = ref[:, j] >= 0
        mode = np.bincount(ref[obs, j]).argmax()
        v = np.where(X[:, j] >= 0, X[:, j], mode)
        for s in range(len(STATES[NODES[j]])):
            cols.append((v == s).astype(float))
    return np.column_stack(cols)


def score(preds):
    """AUC, log loss, Brier and skill versus the prevalence-only forecast, per method with 95% intervals over repeats."""
    rows = []
    for m in ("bn_em", "bn_deletion", "logistic", "prevalence"):
        per = []
        for r, g in preds.groupby("repeat"):
            y, p = g.y.values, np.clip(g[m].values, 1e-6, 1 - 1e-6)
            ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            br = np.mean((p - y) ** 2)
            per.append((_auc(y, p), ll, br))
        a = np.array(per)
        base = preds.groupby("repeat").apply(lambda g: np.mean((g.prevalence - g.y) ** 2), include_groups=False).values
        rows.append(dict(method=m, auc=np.nanmean(a[:, 0]), auc_sd=np.nanstd(a[:, 0]), logloss=a[:, 1].mean(),
                         logloss_sd=a[:, 1].std(), brier=a[:, 2].mean(), brier_sd=a[:, 2].std(),
                         brier_skill=float(1 - (a[:, 2] / base).mean())))
    return pd.DataFrame(rows)


def calibration(preds, method="bn_em", bins=5):
    """Pooled reliability table with quantile bins (there are only 24 positives, so bins stay wide)."""
    p = preds.groupby("id").agg(y=("y", "first"), p=(method, "mean")).reset_index()
    p["bin"] = pd.qcut(p.p.rank(method="first"), bins, labels=False)
    t = p.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"),
                              positives=("y", "sum")).reset_index()
    ece = float((t.n / t.n.sum() * (t.mean_predicted - t.observed_rate).abs()).sum())
    return t, ece


def choose_strength(d, grid=(2, 5, 10, 20, 50, 100, 200), repeats=5):
    out = []
    for s in grid:
        sc = score(cross_validate(d, strength=s, repeats=repeats)).set_index("method").loc["bn_em"]
        out.append(dict(strength=s, logloss=sc.logloss, brier=sc.brier, auc=sc.auc))
    return pd.DataFrame(out)


# ============================================================================ model checks on the data
def ci_checks(d, net):
    """Test the DAG's implied independencies on the data: stratified chi-square (Cochran-style: statistics and
    degrees of freedom summed over strata). Small p = the data disagree with a missing arrow."""
    rows = []
    for a, b, given in net.implied_independencies():
        if a == "ethnicity" or b == "ethnicity":
            pass
        cols = [a, b, *given]
        sub = d[cols].dropna()
        chi, dof, used = 0.0, 0, 0
        keys = sub[list(given)].astype(str).agg("|".join, axis=1) if given else pd.Series("all", index=sub.index)
        for k, g in sub.groupby(keys):
            t = pd.crosstab(g[a], g[b])
            if t.shape[0] < 2 or t.shape[1] < 2 or t.values.sum() < 10:
                continue
            c, _, df_, _ = stats.chi2_contingency(t, correction=False)
            chi += c; dof += df_; used += int(t.values.sum())
        p = float(stats.chi2.sf(chi, dof)) if dof > 0 else np.nan
        rows.append(dict(a=a, b=b, given=", ".join(given), n=used, chi2=chi, dof=dof, p=p))
    return pd.DataFrame(rows)


# ============================================================================ scenarios with bootstrap intervals
SCENARIOS = [
    dict(key="wic_low_vs_high", title="Low-income on WIC vs higher-income without WIC",
         a=dict(label="Low income, on WIC", evidence={"income": "below_poverty", "wic_child": "yes"}),
         b=dict(label="Higher income, no WIC", evidence={"income": "above_185pct", "wic_child": "no"}), do=None),
    dict(key="secure_food", title="Food-insecure child: make the household food secure",
         a=dict(label="Now (insecure)", evidence={"income": "below_poverty", "food_security": "insecure"}),
         b=dict(label="After do(food secure)", evidence={"income": "below_poverty"}, do={"food_security": "secure"}), do=None),
    dict(key="enrol_wic", title="Below-poverty child not on WIC: enrol in WIC",
         a=dict(label="Now (no WIC)", evidence={"income": "below_poverty", "wic_child": "no"}),
         b=dict(label="After do(WIC)", evidence={"income": "below_poverty"}, do={"wic_child": "yes"}), do=None),
    dict(key="enrol_snap", title="Below-poverty child not on SNAP: enrol in SNAP",
         a=dict(label="Now (no SNAP)", evidence={"income": "below_poverty", "snap_past_year": "no"}),
         b=dict(label="After do(SNAP)", evidence={"income": "below_poverty"}, do={"snap_past_year": "yes"}), do=None),
    dict(key="high_need", title="Highest-need profile: WIC and SNAP together",
         a=dict(label="Now", evidence={"ethnicity": "NHBlack", "income": "below_poverty", "parent_edu": "low",
                                       "wic_child": "no", "snap_past_year": "no", "food_security": "insecure"}),
         b=dict(label="After do(WIC, SNAP)", evidence={"ethnicity": "NHBlack", "income": "below_poverty", "parent_edu": "low"},
                do={"wic_child": "yes", "snap_past_year": "yes"}), do=None),
    dict(key="revision", title="Probability revision: evidence arriving one item at a time",
         steps=[("No evidence", {}), ("+ ethnicity: Other Hispanic", {"ethnicity": "OthHisp"}),
                ("+ income: below poverty", {"ethnicity": "OthHisp", "income": "below_poverty"}),
                ("+ food security: insecure", {"ethnicity": "OthHisp", "income": "below_poverty", "food_security": "insecure"}),
                ("+ WIC: yes", {"ethnicity": "OthHisp", "income": "below_poverty", "food_security": "insecure", "wic_child": "yes"})]),
]


def _q(net, spec):
    return float(net.ve_query("anaemia", spec.get("evidence"), spec.get("do"))[1])


def bootstrap_nets(d, reps=200, strength=STRENGTH, seed=0):
    X = make_net().encode(d)
    base = make_net().fit_em(X, strength=strength)
    rng = np.random.default_rng(seed)
    nets = []
    for _ in range(reps):
        idx = rng.integers(0, len(X), len(X))
        nets.append(make_net().fit_em(X[idx], strength=strength, init=base.cpt))
    return base, nets


def run_scenarios(base, boots):
    rows = []
    for sc in SCENARIOS:
        if "steps" in sc:
            for label, ev in sc["steps"]:
                spec = dict(evidence=ev)
                v = _q(base, spec)
                bs = np.array([_q(b, spec) for b in boots])
                rows.append(dict(scenario=sc["key"], title=sc["title"], arm=label, risk=v,
                                 lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
            continue
        vals = {}
        for arm in ("a", "b"):
            v = _q(base, sc[arm]); bs = np.array([_q(b, sc[arm]) for b in boots])
            vals[arm] = (v, bs)
            rows.append(dict(scenario=sc["key"], title=sc["title"], arm=sc[arm]["label"], risk=v,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
        diff = vals["b"][1] - vals["a"][1]
        rows.append(dict(scenario=sc["key"], title=sc["title"], arm="Change (percentage points)",
                         risk=100 * (vals["b"][0] - vals["a"][0]), lo=100 * np.percentile(diff, 2.5),
                         hi=100 * np.percentile(diff, 97.5)))
    return pd.DataFrame(rows)
