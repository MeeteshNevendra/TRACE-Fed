"""Aggregate results_v2/runs into tables, statistical tests and figures.

Statistics
* mean ± std over seeds (dataset2: over the 5 CV folds, which cover every image once),
* 95 % CI of the seed mean (t distribution),
* paired bootstrap of the macro-F1 difference (proposed - competitor) on per-image
  test predictions pooled over seeds (pairs = same seed, same image),
* Friedman test over (dataset, seed) blocks + average ranks for the FL comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc", "ece", "dice", "iou"]


# ----------------------------------------------------------------------------- loading
def load_central(work: Path) -> pd.DataFrame:
    rows = []
    for f in (work / "runs" / "central").glob("*/*/*/seed*.json"):
        r = json.loads(f.read_text())
        if r["stage"] in {"smoke"}:
            continue
        row = {"stage": r["stage"], "dataset": r["dataset"], "model": r["model"], "seed": r["seed"], "fold": r["fold"],
               "params_M": r["params_total"] / 1e6, "trainable_M": r["params_trainable"] / 1e6,
               "best_epoch": r["best_epoch"], "train_min": r["train_time_sec"] / 60, "path": str(f)}
        row.update({f"val_{k}": r["val"].get(k) for k in ("f1", "accuracy")})
        for k in METRICS:
            row[k] = r.get("test", {}).get(k, np.nan)
        if r.get("gate_means"):
            row["gate_mean"] = float(np.mean([g for g in r["gate_means"] if g is not None] or [np.nan]))
        for name, m in (r.get("cross_test") or {}).items():
            row[f"cross_{name}_f1"] = m["f1"]
            row[f"cross_{name}_auc"] = m["auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_fl(work: Path) -> pd.DataFrame:
    rows = []
    for f in (work / "runs" / "fl").glob("*/*/*/*/seed*.json"):
        r = json.loads(f.read_text())
        if r["stage"] == "smoke":
            continue
        setting = f.parent.parent.name
        for test_name, m in r["test"].items():
            row = {"stage": r["stage"], "scenario": r["scenario"], "dataset": r["dataset"], "setting": setting,
                   "tag": r["tag"], "method": r["method"], "seed": r["seed"], "test_set": test_name,
                   "best_round": r["best_round"], "client_val_f1": r["best_client_val_f1"],
                   "uplink_MB": r["uplink_mb_total"], "downlink_MB": r["downlink_mb_total"],
                   "shared_M": r["shared_params"] / 1e6, "train_min": r["train_time_sec"] / 60, "path": str(f)}
            row.update({k: m.get(k, np.nan) for k in METRICS})
            if r["config"].get("noisy_clients"):
                w = [c["weight"] for c in r["history"][r["best_round"] - 1]["clients"] if c["noisy"]]
                row["noisy_client_weight"] = float(np.mean(w)) if w else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def mean_ci(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m, s = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
    h = stats.t.ppf(0.975, len(x) - 1) * s / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, s, h


def summarize(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n_seeds"] = g["seed"].nunique()
        for m in metrics:
            if m in g:
                mu, sd, h = mean_ci(g[m])
                row[f"{m}_mean"], row[f"{m}_std"], row[f"{m}_ci95"] = mu, sd, h
        out.append(row)
    return pd.DataFrame(out)


def _load_preds(path: str, prefix: str = "") -> dict | None:
    p = Path(path).with_suffix(".npz")
    if not p.exists():
        return None
    z = np.load(p)
    if prefix:
        return {k: z[f"{prefix}_{k}"] for k in ("probs", "labels", "uids") if f"{prefix}_{k}" in z}
    return {k: z[k] for k in ("probs", "labels", "uids")}


def paired_bootstrap(rows_a: pd.DataFrame, rows_b: pd.DataFrame, prefix_col: str | None = None, n_boot: int = 5000,
                     seed: int = 0) -> dict:
    """Paired bootstrap over (seed, image) pairs of the macro-F1 difference a - b."""
    ya, pa, pb = [], [], []
    for s in sorted(set(rows_a.seed) & set(rows_b.seed)):
        ra, rb = rows_a[rows_a.seed == s].iloc[0], rows_b[rows_b.seed == s].iloc[0]
        A = _load_preds(ra.path, ra[prefix_col] if prefix_col else "")
        B = _load_preds(rb.path, rb[prefix_col] if prefix_col else "")
        if not A or not B or not np.array_equal(A["uids"], B["uids"]):
            continue
        ya.append(A["labels"] + 1000 * s)  # keep seeds apart for stratified resampling
        pa.append(A["probs"].argmax(1))
        pb.append(B["probs"].argmax(1))
    if not ya:
        return {"delta_f1": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    seeds_lab = np.concatenate(ya)
    y = seeds_lab % 1000
    a, b = np.concatenate(pa), np.concatenate(pb)
    groups = seeds_lab // 1000

    def f1_by_seed(idx):
        vals_a, vals_b = [], []
        for g in np.unique(groups):
            sel = idx[groups[idx] == g]
            vals_a.append(f1_score(y[sel], a[sel], average="macro", zero_division=0))
            vals_b.append(f1_score(y[sel], b[sel], average="macro", zero_division=0))
        return np.mean(vals_a) - np.mean(vals_b)

    rng = np.random.default_rng(seed)
    base = f1_by_seed(np.arange(len(y)))
    boots = np.empty(n_boot)
    by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in by_group])
        boots[i] = f1_by_seed(idx)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta_f1": base, "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5), "p_value": min(1.0, p)}


def friedman(df: pd.DataFrame, method_col: str, block_cols: list[str], metric: str = "f1") -> dict:
    pivot = df.pivot_table(index=block_cols, columns=method_col, values=metric).dropna()
    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {}
    stat, p = stats.friedmanchisquare(*[pivot[c] for c in pivot.columns])
    ranks = pivot.rank(axis=1, ascending=False).mean().sort_values()
    k, n = pivot.shape[1], pivot.shape[0]
    q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}.get(k, np.nan)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    return {"chi2": stat, "p_value": p, "blocks": n, "methods": k, "avg_rank": ranks.to_dict(), "nemenyi_cd": cd}


# ----------------------------------------------------------------------------- tables
def fmt(mu, sd, digits=3):
    return "—" if pd.isna(mu) else f"{mu:.{digits}f} ± {sd:.{digits}f}"


def to_markdown(df: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def pretty(summary: pd.DataFrame, id_cols: list[str], metrics: list[str], extra: list[str] = ()) -> pd.DataFrame:
    out = summary[id_cols + list(extra)].copy()
    for m in metrics:
        if f"{m}_mean" in summary:
            out[m] = [fmt(a, b) for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("results_v2"))
    args = ap.parse_args()
    work = args.work
    tdir, fdir = work / "tables", work / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    sel = json.loads((work / "selection.json").read_text()) if (work / "selection.json").exists() else {}
    proposed_central = sel.get("central", {}).get("trace", {}).get("spec", "trace")

    central = load_central(work)
    fl = load_fl(work)
    central.to_csv(tdir / "all_central_runs.csv", index=False)
    fl.to_csv(tdir / "all_fl_runs.csv", index=False)
    stat_rows = []

    # --- search (validation only)
    s = central[central.stage == "search"]
    if len(s):
        tab = summarize(s, ["dataset", "model"], ["val_f1", "val_accuracy"]).sort_values("val_f1_mean", ascending=False)
        tab.to_csv(tdir / "search_central_validation.csv", index=False)
        to_markdown(pretty(tab, ["dataset", "model"], ["val_f1", "val_accuracy"]), tdir / "search_central_validation.md",
                    "Model search (validation macro-F1, search seeds)")

    # --- centralized final comparison
    c = central[central.stage == "final"]
    if len(c):
        tab = summarize(c, ["dataset", "model"], METRICS + ["params_M", "train_min"])
        tab = tab.sort_values(["dataset", "f1_mean"], ascending=[True, False])
        tab.to_csv(tdir / "central_comparison.csv", index=False)
        to_markdown(pretty(tab, ["dataset", "model"], ["accuracy", "balanced_accuracy", "f1", "auc", "ece", "dice"], ["params_M_mean"]),
                    tdir / "central_comparison.md", "Centralized comparison (test, mean ± std over 5 seeds / folds)")
        for ds, g in c.groupby("dataset"):
            prop = g[g.model == proposed_central]
            for other, go in g[g.model != proposed_central].groupby("model"):
                r = paired_bootstrap(prop, go)
                stat_rows.append({"comparison": "central", "dataset": ds, "proposed": proposed_central, "competitor": other, **r})

    # --- ablations (central)
    a = pd.concat([central[central.stage == "ablation"], c[c.model == proposed_central]])
    if len(central[central.stage == "ablation"]):
        tab = summarize(a, ["dataset", "model"], ["f1", "accuracy", "auc", "dice", "gate_mean"]).sort_values(["dataset", "f1_mean"], ascending=[True, False])
        tab.to_csv(tdir / "ablation_central.csv", index=False)
        to_markdown(pretty(tab, ["dataset", "model"], ["f1", "accuracy", "auc", "dice"]), tdir / "ablation_central.md",
                    "TRACE-Net ablations (test, 5 seeds)")

    # --- federated main
    f = fl[(fl.stage == "final") & (fl.scenario == "main")]
    if len(f):
        tab = summarize(f, ["dataset", "setting", "tag"], METRICS + ["uplink_MB", "downlink_MB", "shared_M"])
        tab = tab.sort_values(["dataset", "f1_mean"], ascending=[True, False])
        tab.to_csv(tdir / "fl_comparison.csv", index=False)
        to_markdown(pretty(tab, ["dataset", "tag"], ["accuracy", "balanced_accuracy", "f1", "auc", "ece"], ["uplink_MB_mean", "shared_M_mean"]),
                    tdir / "fl_comparison.md", "Federated comparison (Dirichlet α = 0.5, 5 clients, test, 5 seeds)")
        for ds, g in f.groupby("dataset"):
            prop = g[g.tag == "trace"]
            for other, go in g[g.tag != "trace"].groupby("tag"):
                r = paired_bootstrap(prop, go, prefix_col="test_set")
                stat_rows.append({"comparison": "fl", "dataset": ds, "proposed": "trace", "competitor": other, **r})
        fr = friedman(f, "tag", ["dataset", "seed"])
        if fr:
            (tdir / "fl_friedman.json").write_text(json.dumps(fr, indent=2, default=float))
        _fig_f1_vs_comm(f, fdir / "fl_f1_vs_uplink.png")
        _fig_convergence(f, fdir / "fl_convergence.png")

    # --- FL ablations / sweeps / robustness / mixed
    for stage, name in [("ablation", "ablation_fl"), ("hetero", "fl_heterogeneity"), ("clients", "fl_clients"), ("robust", "fl_robustness")]:
        g = fl[fl.stage == stage]
        if stage == "ablation":
            if "drift=none" in sel.get("fl", {}).get("trace", {}).get("opts", ""):
                # selected TRACE-Fed uses no drift correction, so "no drift" duplicates the full method
                g = g[g.tag != "trace_no_drift"]
            g = pd.concat([g, f[f.tag == "trace"]])
        if len(g):
            cols = METRICS + ["uplink_MB"] + (["noisy_client_weight"] if stage == "robust" else [])
            tab = summarize(g, ["dataset", "setting", "tag"], cols).sort_values(["dataset", "setting", "f1_mean"], ascending=[True, True, False])
            tab.to_csv(tdir / f"{name}.csv", index=False)
            to_markdown(pretty(tab, ["dataset", "setting", "tag"], ["f1", "accuracy", "auc"], ["uplink_MB_mean"]), tdir / f"{name}.md", name)
    m = fl[fl.scenario == "mixed"]
    if len(m):
        tab = summarize(m, ["test_set", "tag"], METRICS + ["uplink_MB"]).sort_values(["test_set", "f1_mean"], ascending=[True, False])
        tab.to_csv(tdir / "fl_mixed_sites.csv", index=False)
        to_markdown(pretty(tab, ["test_set", "tag"], ["f1", "accuracy", "auc"], ["uplink_MB_mean"]), tdir / "fl_mixed_sites.md",
                    "Mixed-site federation (binary)")

    # --- cross-dataset
    x = central[central.stage == "cross"]
    if len(x):
        cross_cols = [col for col in x.columns if col.startswith("cross_") and col.endswith("_f1")]
        long = x.melt(id_vars=["dataset", "model", "seed", "f1"], value_vars=cross_cols, var_name="target", value_name="cross_f1").dropna()
        long["target"] = long["target"].str.replace("cross_", "").str.replace(r"_fold\d+_f1", "", regex=True)
        tab = long.groupby(["dataset", "target", "model"]).agg(in_domain_f1=("f1", "mean"), cross_f1=("cross_f1", "mean"),
                                                                cross_f1_std=("cross_f1", "std")).reset_index()
        tab.to_csv(tdir / "cross_dataset.csv", index=False)
        to_markdown(tab.round(4), tdir / "cross_dataset.md", "Cross-dataset binary generalisation")

    if stat_rows:
        st = pd.DataFrame(stat_rows)
        # Holm correction within each comparison family
        st["p_holm"] = np.nan
        for (cmp, ds), g in st.groupby(["comparison", "dataset"]):
            order = g["p_value"].dropna().sort_values().index  # personalised methods have no shared predictions
            m_ = len(order)
            adj, running = {}, 0.0
            for i, idx in enumerate(order):
                running = max(running, min(1.0, (m_ - i) * st.loc[idx, "p_value"]))
                adj[idx] = running
            st.loc[list(adj), "p_holm"] = pd.Series(adj)
        st.to_csv(tdir / "significance_paired_bootstrap.csv", index=False)
        to_markdown(st.round(4), tdir / "significance_paired_bootstrap.md", "Paired bootstrap (macro-F1 difference, Holm-adjusted)")
    print(f"tables -> {tdir}\nfigures -> {fdir}")


# ----------------------------------------------------------------------------- figures
def _fig_f1_vs_comm(f: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(f.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = f[f.dataset == ds].groupby("tag").agg(f1=("f1", "mean"), f1_sd=("f1", "std"), up=("uplink_MB", "mean"))
        for tag, r in g.iterrows():
            is_p = tag == "trace"
            ax.errorbar(r.up, r.f1, yerr=r.f1_sd, fmt="o", ms=8 if is_p else 5, color="#c0392b" if is_p else "#34495e", capsize=3)
            ax.annotate("TRACE-Fed" if is_p else tag, (r.up, r.f1), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("total uplink (MB, log scale)")
        ax.set_ylabel("test macro-F1")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fig_convergence(f: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(f.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for tag, g in f[(f.dataset == ds)].drop_duplicates("path").groupby("tag"):
            curves = [[h["val_f1_clients"] for h in json.loads(Path(p).read_text())["history"]] for p in g.path]
            L = min(map(len, curves))
            arr = np.array([c[:L] for c in curves])
            is_p = tag == "trace"
            ax.plot(np.arange(1, L + 1), arr.mean(0), lw=2.2 if is_p else 1.0, color="#c0392b" if is_p else None,
                    label="TRACE-Fed" if is_p else tag)
        ax.set_xlabel("round")
        ax.set_ylabel("client-validation macro-F1")
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    axes[0, -1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
