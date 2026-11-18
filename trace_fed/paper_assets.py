"""Generate every manuscript number, table and figure from results_v2/.

Outputs (paper_v2/):
  generated/numbers.tex   \newcommand macros for every number quoted in the text
  generated/tab_*.tex     LaTeX tables
  figures/*.pdf|png       figures
Run: python -m trace_fed.paper_assets
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .report import load_central, load_fl

WORK = Path("results_v2")
OUT = Path("paper_v2")
GEN = OUT / "generated"
FIG = OUT / "figures"

MODEL_NAMES = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-T", "vit_small": "ViT-S/16", "swin_tiny": "Swin-T", "trace": "TRACE-Net",
}
FL_NAMES = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fedadam": "FedAdam", "fedbn": "FedBN",
    "fedper": "FedPer", "fedrep": "FedRep", "moon": "MOON", "trace": "TRACE-Fed (ours)",
}
FL_ORDER = ["fedavg", "fedprox", "scaffold", "fedadam", "moon", "fedbn", "fedper", "fedrep", "trace"]
CENTRAL_ORDER = ["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny", "vit_small", "swin_tiny", "trace"]
PERSONALISED = {"fedbn", "fedper", "fedrep"}

macros: dict[str, str] = {}


def mac(name: str, value) -> None:
    assert name.isalpha(), name
    macros[name] = str(value)


def f3(x) -> str:
    return "--" if pd.isna(x) else f"{x:.3f}"


def pm(mu, sd, d=3) -> str:
    return "--" if pd.isna(mu) else f"{mu:.{d}f}$\\pm${sd:.{d}f}"


def base(spec: str) -> str:
    return spec.split(":")[0]


def write(name: str, text: str) -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    if name.startswith("tab_") and name != "tab_ranks.tex":  # scale every table to the column/page width
        text = text.replace(r"\begin{tabular}", r"\resizebox{\linewidth}{!}{\begin{tabular}").replace(r"\end{tabular}", r"\end{tabular}}")
    (GEN / name).write_text(text, encoding="utf-8")


def sig_marker(st: pd.DataFrame, cmp: str, ds: str, competitor_base: str) -> str:
    """dagger: proposed significantly better; ddagger: significantly worse (Holm p<0.05)."""
    rows = st[(st.comparison == cmp) & (st.dataset == ds)]
    rows = rows[rows.competitor.map(base) == competitor_base]
    if rows.empty or pd.isna(rows.iloc[0].p_holm):
        return ""
    r = rows.iloc[0]
    if r.p_holm < 0.05:
        return "$^{\\dagger}$" if r.delta_f1 > 0 else "$^{\\ddagger}$"
    return ""


def bold_best(values: list[float], higher=True) -> int:
    arr = np.array([v if not pd.isna(v) else (-np.inf if higher else np.inf) for v in values])
    return int(arr.argmax() if higher else arr.argmin())


# =========================================================================== data audit
def data_audit() -> None:
    s = json.loads((WORK / "data_audit" / "summary.json").read_text())
    d = pd.read_csv(WORK / "data_audit" / "duplicate_groups.csv")
    from .data import build_catalog

    cat = build_catalog(Path("."), WORK / "data_audit" / "duplicate_groups.csv")
    d1_orig = s["images_per_split"]["dataset1/train"] + s["images_per_split"]["dataset1/val"]
    d2_orig = s["images_per_split"]["dataset2/train"] + s["images_per_split"]["dataset2/val"]
    d1_n = int((cat.dataset == "dataset1").sum())
    d2_n = int((cat.dataset == "dataset2").sum())
    exact_groups = d[d.exact].group.nunique()
    val_leak = d[(d.group_cross_split) & (d.dataset == "dataset1") & (d.split == "val")].shape[0]
    mac("DOneOrig", f"{d1_orig:,}"); mac("DTwoOrig", d2_orig)
    mac("DOneUnique", f"{d1_n:,}"); mac("DTwoUnique", d2_n)
    mac("DupGroups", f"{s['duplicate_groups']:,}"); mac("DupExactGroups", f"{exact_groups:,}")
    mac("DupRedundant", f"{s['redundant_images_removable']:,}")
    mac("DupRedundantPct", f"{100 * s['redundant_images_removable'] / s['images_total']:.1f}")
    mac("DupCrossSplit", s["groups_crossing_splits"])
    mac("ValLeak", val_leak); mac("ValLeakPct", f"{100 * val_leak / s['images_per_split']['dataset1/val']:.0f}")
    mac("DOneValOrig", s["images_per_split"]["dataset1/val"])
    nt_orig = 711 + 100  # No Tumor images in Train + Val folders
    nt_new = int(((cat.dataset == "dataset1") & (cat.class_name == "No Tumor")).sum())
    mac("NoTumorOrig", nt_orig); mac("NoTumorUnique", nt_new)
    mac("BoxMismatch", int(cat.box_class_mismatch.sum()))
    mac("BoxMismatchPct", f"{100 * cat.box_class_mismatch.sum() / ((cat.dataset == 'dataset1') & (cat.class_name != 'No Tumor')).sum():.1f}")
    mac("NearThreshold", s["near_threshold_bits_of_256"])

    sp = pd.read_csv(WORK / "splits" / "split_summary.csv")
    rows = []
    for ds, name, classes in [("dataset1", "D1 (4-class)", ["Glioma", "Meningioma", "No Tumor", "Pituitary"]),
                              ("dataset2", "D2 (binary)", ["Normal", "Tumor"])]:
        orig = d1_orig if ds == "dataset1" else d2_orig
        uniq = d1_n if ds == "dataset1" else d2_n
        counts = cat[cat.dataset == ds].class_name.value_counts()
        g = sp[(sp.dataset == ds) & (sp.fold == 0)].set_index("split").n
        split_txt = f"{g['train']}/{g['val']}/{g['test']}" + (" per fold" if ds == "dataset2" else "")
        rows.append(f"{name} & {orig:,} & {orig - uniq:,} & {uniq:,} & " + ", ".join(f"{c}: {counts[c]}" for c in classes) +
                    f" & {split_txt} \\\\")
    write("tab_data.tex", "\n".join([
        r"\begin{table*}[t]\centering\caption{Datasets after the duplicate audit. Duplicate groups (exact copies and near-copies at "
        r"$\le$\NearThreshold{}/256 differing dHash bits) are collapsed to one image before splitting. D1 uses one fixed stratified 70/15/15 split; "
        r"D2 uses stratified 5-fold cross-validation (15\% of each training fold held out for validation).}\label{tab:data}",
        r"\begin{tabular}{lrrrll}\toprule",
        r"Dataset & Images & Removed duplicates & Unique & Unique images per class & Train/val/test \\ \midrule",
        *rows, r"\bottomrule\end{tabular}\end{table*}"]))
    src = WORK / "data_audit" / "near_duplicate_samples.png"
    if src.exists():
        shutil.copy(src, FIG / "near_duplicates.png")


# =========================================================================== centralized
def central_tables(c: pd.DataFrame, st: pd.DataFrame) -> None:
    cf = c[c.stage == "final"].copy()
    cf["base"] = cf.model.map(base)
    lines = [r"\begin{table*}[t]\centering",
             r"\caption{Centralized test performance (mean$\pm$std over 5 seeds on D1 and over the 5 cross-validation folds on D2). "
             r"Macro-averaged metrics; ECE = expected calibration error (lower is better). $^{\dagger}$/$^{\ddagger}$: TRACE-Net is significantly "
             r"better/worse than this model (paired bootstrap of macro-F1 over images and seeds, Holm-corrected $p<0.05$). Best mean in bold.}"
             r"\label{tab:central}",
             r"\begin{tabular}{lr ccccc ccccc}\toprule",
             r" & & \multicolumn{5}{c}{D1: 4-class (test $n$=548)} & \multicolumn{5}{c}{D2: binary (all 395 images over 5 folds)} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
             r"Model & Params (M) & Acc & Bal.\ Acc & Macro-F1 & AUC & ECE$\downarrow$ & Acc & Bal.\ Acc & Macro-F1 & AUC & ECE$\downarrow$ \\ \midrule"]
    metrics = ["accuracy", "balanced_accuracy", "f1", "auc", "ece"]
    agg = cf.groupby(["dataset", "base"])[metrics + ["params_M"]].agg(["mean", "std"])
    best = {}
    for ds in ["dataset1", "dataset2"]:
        for m in metrics:
            vals = [agg.loc[(ds, b), (m, "mean")] for b in CENTRAL_ORDER]
            best[(ds, m)] = CENTRAL_ORDER[bold_best(vals, higher=(m != "ece"))]
    for b in CENTRAL_ORDER:
        cells = [MODEL_NAMES[b] + (r" (ours)" if b == "trace" else ""), f"{agg.loc[('dataset1', b), ('params_M', 'mean')]:.1f}"]
        for ds in ["dataset1", "dataset2"]:
            for m in metrics:
                txt = pm(agg.loc[(ds, b), (m, "mean")], agg.loc[(ds, b), (m, "std")])
                if best[(ds, m)] == b:
                    txt = r"\textbf{" + txt + "}"
                if m == "f1" and b != "trace":
                    txt += sig_marker(st, "central", ds, b)
                cells.append(txt)
        if b == "trace":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table*}"]
    write("tab_central.tex", "\n".join(lines))

    t = agg.xs("trace", level="base")
    for ds, key in [("dataset1", "One"), ("dataset2", "Two")]:
        mac(f"CenTraceFOne{key}" if False else f"CenTraceF{key}", f3(t.loc[ds, ("f1", "mean")]))
        mac(f"CenTraceFSd{key}", f3(t.loc[ds, ("f1", "std")]))
        mac(f"CenTraceAuc{key}", f3(t.loc[ds, ("auc", "mean")]))
        bb = agg.loc[ds].drop("trace")[("f1", "mean")].idxmax()
        mac(f"CenBestBase{key}", MODEL_NAMES[bb])
        mac(f"CenBestBaseF{key}", f3(agg.loc[(ds, bb), ("f1", "mean")]))
        mac(f"CenBestBaseFSd{key}", f3(agg.loc[(ds, bb), ("f1", "std")]))
        rank = int((agg.loc[ds][("f1", "mean")] > t.loc[ds, ("f1", "mean")]).sum()) + 1
        mac(f"CenTraceRank{key}", rank)
    mac("TraceParams", f"{agg.loc[('dataset1', 'trace'), ('params_M', 'mean')]:.1f}")
    mac("ConvnextParams", f"{agg.loc[('dataset1', 'convnext_tiny'), ('params_M', 'mean')]:.1f}")
    mac("SwinParams", f"{agg.loc[('dataset1', 'swin_tiny'), ('params_M', 'mean')]:.1f}")
    mac("CenTraceDiceOne", f3(cf[(cf.base == "trace") & (cf.dataset == "dataset1")].dice.mean()))
    mac("CenTraceIoUOne", f3(cf[(cf.base == "trace") & (cf.dataset == "dataset1")].iou.mean()))
    g = cf[cf.base == "trace"].gate_mean
    mac("GateMeanOne", f"{cf[(cf.base == 'trace') & (cf.dataset == 'dataset1')].gate_mean.mean():.2f}")
    mac("GateMeanTwo", f"{cf[(cf.base == 'trace') & (cf.dataset == 'dataset2')].gate_mean.mean():.2f}")
    _ = g
    # significance summary for the text
    s1 = st[(st.comparison == "central")]
    sig_better = s1[(s1.p_holm < 0.05) & (s1.delta_f1 > 0)]
    mac("CenSigBetterTwo", ", ".join(MODEL_NAMES[base(x)] for x in sig_better[sig_better.dataset == "dataset2"].competitor))
    mac("CenSigBetterCountTwo", len(sig_better[sig_better.dataset == "dataset2"]))
    mac("CenMinPOne", f"{s1[s1.dataset == 'dataset1'].p_holm.min():.2f}")
    sw = s1[(s1.dataset == "dataset1") & (s1.competitor.map(base) == "swin_tiny")].iloc[0]
    mac("CenSwinDeltaOne", f"{-sw.delta_f1:.3f}")
    mac("CenSwinPOne", f"{sw.p_value:.2f}")


# =========================================================================== federated
def fl_tables(f: pd.DataFrame, st: pd.DataFrame) -> None:
    ff = f[(f.stage == "final") & (f.scenario == "main")]
    metrics = ["accuracy", "f1", "auc", "ece"]
    agg = ff.groupby(["dataset", "tag"])[metrics + ["uplink_MB", "downlink_MB", "shared_M"]].agg(["mean", "std"])
    lines = [r"\begin{table*}[t]\centering",
             r"\caption{Federated test performance under Dirichlet ($\alpha{=}0.5$) label skew with 5 clients and 30 rounds "
             r"(mean$\pm$std over 5 seeds; D2 seeds use the 5 cross-validation folds). Communication is the total over all clients and rounds. "
             r"All baselines federate the full TRACE-Net (14.9\,M parameters); FedBN/FedPer/FedRep are evaluated with each client's personal "
             r"parameters and averaged over clients. $^{\dagger}$/$^{\ddagger}$: TRACE-Fed significantly better/worse "
             r"(paired bootstrap, Holm-corrected $p<0.05$; not defined for personalised methods). Best mean in bold.}\label{tab:fl}",
             r"\begin{tabular}{l rrr ccc ccc}\toprule",
             r" & \multicolumn{3}{c}{Communication} & \multicolumn{3}{c}{D1: 4-class} & \multicolumn{3}{c}{D2: binary} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Method & Shared (M) & Up (MB) & Down (MB) & Acc & Macro-F1 & ECE$\downarrow$ & Acc & Macro-F1 & ECE$\downarrow$ \\ \midrule"]
    best = {}
    for ds in ["dataset1", "dataset2"]:
        for m in ["accuracy", "f1", "ece"]:
            vals = [agg.loc[(ds, t), (m, "mean")] for t in FL_ORDER]
            best[(ds, m)] = FL_ORDER[bold_best(vals, higher=(m != "ece"))]
    for t in FL_ORDER:
        r1 = agg.loc[("dataset1", t)]
        cells = [FL_NAMES[t], f"{r1[('shared_M', 'mean')]:.2f}", f"{r1[('uplink_MB', 'mean')]:,.0f}", f"{r1[('downlink_MB', 'mean')]:,.0f}"]
        for ds in ["dataset1", "dataset2"]:
            for m in ["accuracy", "f1", "ece"]:
                txt = pm(agg.loc[(ds, t), (m, "mean")], agg.loc[(ds, t), (m, "std")])
                if best[(ds, m)] == t:
                    txt = r"\textbf{" + txt + "}"
                if m == "f1" and t != "trace":
                    txt += sig_marker(st, "fl", ds, t)
                cells.append(txt)
        if t == "trace":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}\end{table*}")
    write("tab_fl.tex", "\n".join(lines))

    for ds, key in [("dataset1", "One"), ("dataset2", "Two")]:
        mac(f"FlTraceF{key}", f3(agg.loc[(ds, "trace"), ("f1", "mean")]))
        mac(f"FlTraceFSd{key}", f3(agg.loc[(ds, "trace"), ("f1", "std")]))
        others = agg.loc[ds].drop("trace")
        bb = others[("f1", "mean")].idxmax()
        mac(f"FlBest{key}", FL_NAMES[bb]); mac(f"FlBestF{key}", f3(others.loc[bb, ("f1", "mean")]))
        mac(f"FlAvgF{key}", f3(agg.loc[(ds, "fedavg"), ("f1", "mean")]))
        mac(f"FlGapBest{key}", f3(others.loc[bb, ("f1", "mean")] - agg.loc[(ds, "trace"), ("f1", "mean")]))
    up_t = agg.loc[("dataset1", "trace"), ("uplink_MB", "mean")]
    dn_t = agg.loc[("dataset1", "trace"), ("downlink_MB", "mean")]
    up_a = agg.loc[("dataset1", "fedavg"), ("uplink_MB", "mean")]
    mac("UpTrace", f"{up_t:.0f}"); mac("DownTrace", f"{dn_t:.0f}"); mac("UpAvg", f"{up_a:,.0f}")
    mac("UpRatio", f"{up_a / up_t:.0f}"); mac("DownRatio", f"{up_a / dn_t:.0f}")
    mac("UpPct", f"{100 * up_t / up_a:.1f}"); mac("TotalRatio", f"{2 * up_a / (up_t + dn_t):.0f}")
    mac("SharedTrace", f"{agg.loc[('dataset1', 'trace'), ('shared_M', 'mean')]:.2f}")
    mac("SharedFull", f"{agg.loc[('dataset1', 'fedavg'), ('shared_M', 'mean')]:.2f}")
    s = st[st.comparison == "fl"]
    s1 = s[(s.dataset == "dataset1") & s.p_holm.notna()]
    mac("FlMinPOne", f"{s1.p_value.min():.2f}")
    worse2 = s[(s.dataset == "dataset2") & (s.p_holm < 0.05) & (s.delta_f1 < 0)]
    mac("FlWorseTwo", ", ".join(FL_NAMES[x] for x in worse2.competitor))
    fr = json.loads((WORK / "tables" / "fl_friedman.json").read_text())
    mac("FriedmanChi", f"{fr['chi2']:.1f}"); mac("FriedmanP", _sci(fr["p_value"]))
    mac("FriedmanCD", f"{fr['nemenyi_cd']:.2f}"); mac("FriedmanBlocks", fr["blocks"])
    ranks = fr["avg_rank"]
    mac("RankTrace", f"{ranks['trace']:.2f}"); mac("RankBest", f"{min(ranks.values()):.2f}")
    mac("RankBestName", FL_NAMES[min(ranks, key=ranks.get)])
    mac("RankTraceGap", f"{ranks['trace'] - min(ranks.values()):.2f}")
    write("tab_ranks.tex", " & ".join(f"{FL_NAMES[k].replace(' (ours)', '')}: {v:.2f}" for k, v in sorted(ranks.items(), key=lambda kv: kv[1])))


def _sci(p: float) -> str:
    e = int(np.floor(np.log10(p)))
    return f"{p / 10 ** e:.1f}\\times10^{{{e}}}"


# =========================================================================== ablations
def ablation_tables(c: pd.DataFrame, f: pd.DataFrame) -> None:
    cf = c[c.stage == "final"]
    a = pd.concat([c[c.stage == "ablation"], cf[cf.model.map(base) == "trace"]])
    variant_names = {
        "": "TRACE-Net (full)", "mixer=ssm": "SSM branch only", "mixer=attn": "Attention branch only",
        "mixer=sum": "Fixed 0.5/0.5 mixing (no gate)", "pooling=gap": "No mask-guided pooling",
        "loc_head=0": "No localisation head", "xc=0": "No explanation-consistency loss",
        "seg=0,xc=0": "No box supervision ($\\lambda_{seg}{=}\\lambda_{xc}{=}0$)",
    }

    def vkey(spec):
        rest = spec.split(":", 1)[1] if ":" in spec else ""
        parts = [p for p in rest.split(",") if p and not p.startswith("backbone_lr")]
        return ",".join(parts)

    a = a.assign(v=a.model.map(vkey))
    g = a.groupby(["v", "dataset"])[["f1", "dice"]].agg(["mean", "std"])
    lines = [r"\begin{table}[t]\centering\caption{TRACE-Net component ablations (centralized, test, 5 seeds/folds). "
             r"Dice of the predicted lesion map against D1 tumour boxes.}\label{tab:abl_central}",
             r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lccc}\toprule",
             r"Variant & D1 Macro-F1 & D2 Macro-F1 & D1 Dice \\ \midrule"]
    for k, name in variant_names.items():
        if (k, "dataset1") not in g.index:
            continue
        d1 = g.loc[(k, "dataset1")]
        d2 = g.loc[(k, "dataset2")]
        lines.append(f"{name} & {pm(d1[('f1', 'mean')], d1[('f1', 'std')])} & {pm(d2[('f1', 'mean')], d2[('f1', 'std')])} & "
                     f"{pm(d1[('dice', 'mean')], d1[('dice', 'std')]) if not pd.isna(d1[('dice', 'mean')]) else '--'} \\\\")
        if k == "":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_abl_central.tex", "\n".join(lines))
    mac("AblNoBoxDice", f3(g.loc[("seg=0,xc=0", "dataset1"), ("dice", "mean")]))
    mac("AblNoXcF", f3(g.loc[("xc=0", "dataset1"), ("f1", "mean")]))
    mac("AblNoXcDice", f3(g.loc[("xc=0", "dataset1"), ("dice", "mean")]))
    mac("AblAttnF", f3(g.loc[("mixer=attn", "dataset1"), ("f1", "mean")]))
    mac("AblAttnFTwo", f3(g.loc[("mixer=attn", "dataset2"), ("f1", "mean")]))
    f1s = [g.loc[(k, "dataset1"), ("f1", "mean")] for k in variant_names if (k, "dataset1") in g.index]
    mac("AblRangeLo", f3(min(f1s))); mac("AblRangeHi", f3(max(f1s)))

    ff = f[(f.stage == "final") & (f.scenario == "main") & (f.tag == "trace")]
    fa = f[(f.stage == "ablation") & (f.tag != "trace_no_drift")]
    fa = pd.concat([fa, ff])
    names = {"trace": "TRACE-Fed (full)", "trace_no_trust": "No trust weighting", "trace_no_ef": "No error feedback",
             "trace_no_proto": "No prototype alignment", "trace_no_compression": "No compression (dense)",
             "trace_full_model": "Full-model sharing"}
    g = fa.groupby(["tag", "dataset"])[["f1", "uplink_MB"]].agg(["mean", "std"])
    lines = [r"\begin{table}[t]\centering\caption{TRACE-Fed ablations (federated, Dirichlet $\alpha{=}0.5$, 5 clients). Full method: "
             r"5 seeds; variants: 3 seeds. Uplink in MB on D1.}\label{tab:abl_fl}",
             r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lccr}\toprule",
             r"Variant & D1 Macro-F1 & D2 Macro-F1 & Up (MB) \\ \midrule"]
    for k, name in names.items():
        if (k, "dataset1") not in g.index:
            continue
        d1, d2 = g.loc[(k, "dataset1")], g.loc[(k, "dataset2")]
        lines.append(f"{name} & {pm(d1[('f1', 'mean')], d1[('f1', 'std')])} & {pm(d2[('f1', 'mean')], d2[('f1', 'std')])} & "
                     f"{d1[('uplink_MB', 'mean')]:,.0f} \\\\")
        if k == "trace":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_abl_fl.tex", "\n".join(lines))
    for k, n in [("trace_no_trust", "NoTrust"), ("trace_no_ef", "NoEf"), ("trace_no_compression", "NoComp"),
                 ("trace_no_proto", "NoProto"), ("trace_full_model", "FullModel")]:
        mac(f"Fl{n}FOne", f3(g.loc[(k, "dataset1"), ("f1", "mean")]))
        mac(f"Fl{n}FTwo", f3(g.loc[(k, "dataset2"), ("f1", "mean")]))
    mac("FlNoCompUp", f"{g.loc[('trace_no_compression', 'dataset1'), ('uplink_MB', 'mean')]:,.0f}")
    mac("FlFullModelUp", f"{g.loc[('trace_full_model', 'dataset1'), ('uplink_MB', 'mean')]:,.0f}")
    # accidental replicate: identical configuration run twice (see text)
    rep = f[(f.stage == "ablation") & (f.tag == "trace_no_drift")].groupby("dataset").f1.mean()
    mac("ReplicateFTwo", f3(rep.get("dataset2", np.nan)))
    mac("ReplicateFOne", f3(rep.get("dataset1", np.nan)))


# =========================================================================== sweeps, robustness, mixed, cross
def sweep_tables(f: pd.DataFrame) -> None:
    methods = ["fedavg", "fedprox", "scaffold", "trace"]
    base_rows = f[(f.stage == "final") & (f.scenario == "main") & (f.dataset == "dataset1") & f.tag.isin(methods) & (f.seed < 3)]
    h = pd.concat([f[f.stage.isin(["hetero", "clients"])], base_rows])
    g = h.groupby(["setting", "tag"]).f1.agg(["mean", "std"])
    settings = [("iid_c5", "IID, 5 clients"), ("dirichlet1.0_c5", r"$\alpha{=}1.0$, 5 clients"),
                ("dirichlet0.5_c5", r"$\alpha{=}0.5$, 5 clients"), ("dirichlet0.1_c5", r"$\alpha{=}0.1$, 5 clients"),
                ("dirichlet0.5_c3", r"$\alpha{=}0.5$, 3 clients"), ("dirichlet0.5_c10", r"$\alpha{=}0.5$, 10 clients")]
    lines = [r"\begin{table}[t]\centering\caption{Sensitivity to label skew and client count on D1 (macro-F1, mean$\pm$std over 3 seeds). "
             r"Lower $\alpha$ = stronger non-IID skew.}\label{tab:sweep}",
             r"\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lcccc}\toprule",
             "Setting & " + " & ".join(FL_NAMES[m].replace(" (ours)", "") for m in methods) + r" \\ \midrule"]
    for key, name in settings:
        vals = [g.loc[(key, m), "mean"] for m in methods]
        bi = bold_best(vals)
        cells = []
        for i, m in enumerate(methods):
            t = pm(g.loc[(key, m), "mean"], g.loc[(key, m), "std"])
            cells.append(r"\textbf{" + t + "}" if i == bi else t)
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
        if key == "dirichlet0.1_c5":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_sweep.tex", "\n".join(lines))
    for m, n in [("fedavg", "Avg"), ("fedprox", "Prox"), ("scaffold", "Scaf"), ("trace", "Trace")]:
        mac(f"AlphaLow{n}", f3(g.loc[("dirichlet0.1_c5", m), "mean"]))
        mac(f"AlphaLow{n}Sd", f3(g.loc[("dirichlet0.1_c5", m), "std"]))
        mac(f"Ten{n}", f3(g.loc[("dirichlet0.5_c10", m), "mean"]))
    mac("UpTraceTen", f"{h[(h.setting == 'dirichlet0.5_c10') & (h.tag == 'trace')].uplink_MB.mean():.0f}")
    mac("UpAvgTen", f"{h[(h.setting == 'dirichlet0.5_c10') & (h.tag == 'fedavg')].uplink_MB.mean():,.0f}")
    _fig_sweep(g)

    r = f[f.stage == "robust"]
    g = r.groupby("tag")[["f1", "accuracy", "noisy_client_weight"]].agg(["mean", "std"])
    names = {"fedavg": "FedAvg", "fedprox": "FedProx", "trace_no_trust": "TRACE-Fed w/o trust", "trace": "TRACE-Fed"}
    lines = [r"\begin{table}[t]\centering\caption{Robustness to one corrupted client (80\% of its training labels randomised; D1, "
             r"$\alpha{=}0.5$, 5 clients, 3 seeds). Weight = aggregation weight of the corrupted client in the selected round.}\label{tab:robust}",
             r"\begin{tabular}{lcc}\toprule Method & Macro-F1 & Corrupted-client weight \\ \midrule"]
    for k, n in names.items():
        lines.append(f"{n} & {pm(g.loc[k, ('f1', 'mean')], g.loc[k, ('f1', 'std')])} & "
                     f"{g.loc[k, ('noisy_client_weight', 'mean')]:.3f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_robust.tex", "\n".join(lines))
    mac("RobTraceF", f3(g.loc["trace", ("f1", "mean")])); mac("RobNoTrustF", f3(g.loc["trace_no_trust", ("f1", "mean")]))
    mac("RobAvgF", f3(g.loc["fedavg", ("f1", "mean")])); mac("RobProxF", f3(g.loc["fedprox", ("f1", "mean")]))
    mac("RobTraceW", f"{g.loc['trace', ('noisy_client_weight', 'mean')]:.3f}")
    mac("RobAvgW", f"{g.loc['fedavg', ('noisy_client_weight', 'mean')]:.3f}")
    mac("RobGain", f3(g.loc["trace", ("f1", "mean")] - g.loc["trace_no_trust", ("f1", "mean")]))
    _fig_robust(r)

    m = f[f.scenario == "mixed"]
    g = m.groupby(["test_set", "tag"])[["f1", "uplink_MB"]].agg(["mean", "std"])
    meths = ["fedavg", "fedprox", "fedbn", "scaffold", "trace"]
    lines = [r"\begin{table}[t]\centering\caption{Mixed-site federation (binary tumour vs.\ normal): 3 clients drawn from D1 and 2 from D2; "
             r"macro-F1 on each site's test split and pooled (3 seeds).}\label{tab:mixed}",
             r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lcccr}\toprule Method & D1 test & D2 test & Pooled & Up (MB) \\ \midrule"]
    best = {ts: meths[bold_best([g.loc[(ts, x), ("f1", "mean")] for x in meths])] for ts in ["dataset1", "dataset2", "pooled"]}
    for x in meths:
        cells = []
        for ts in ["dataset1", "dataset2", "pooled"]:
            t = pm(g.loc[(ts, x), ("f1", "mean")], g.loc[(ts, x), ("f1", "std")])
            cells.append(r"\textbf{" + t + "}" if best[ts] == x else t)
        lines.append(f"{FL_NAMES[x]} & " + " & ".join(cells) + f" & {g.loc[('pooled', x), ('uplink_MB', 'mean')]:,.0f} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_mixed.tex", "\n".join(lines))
    for ts, n in [("dataset1", "One"), ("dataset2", "Two"), ("pooled", "Pool")]:
        mac(f"MixTrace{n}", f3(g.loc[(ts, "trace"), ("f1", "mean")]))
        mac(f"MixScaf{n}", f3(g.loc[(ts, "scaffold"), ("f1", "mean")]))
        mac(f"MixAvg{n}", f3(g.loc[(ts, "fedavg"), ("f1", "mean")]))


def cross_table(c: pd.DataFrame) -> None:
    x = pd.read_csv(WORK / "tables" / "cross_dataset.csv")
    x["b"] = x.model.map(base)
    lines = [r"\begin{table}[t]\centering\caption{Cross-dataset generalisation of binary (tumour vs.\ normal) models: trained on one "
             r"dataset, tested on the other dataset's held-out test data (macro-F1, 3 seeds).}\label{tab:cross}",
             r"\begin{tabular}{lcc}\toprule Model & D1$\rightarrow$D2 & D2$\rightarrow$D1 \\ \midrule"]
    for b in ["swin_tiny", "trace"]:
        r12 = x[(x.dataset == "dataset1") & (x.b == b)].iloc[0]
        r21 = x[(x.dataset == "dataset2") & (x.b == b)].iloc[0]
        lines.append(f"{MODEL_NAMES[b]} & {pm(r12.cross_f1, r12.cross_f1_std)} & {pm(r21.cross_f1, r21.cross_f1_std)} \\\\")
        key = "Trace" if b == "trace" else "Swin"
        mac(f"Cross{key}OneTwo", f3(r12.cross_f1)); mac(f"Cross{key}TwoOne", f3(r21.cross_f1))
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_cross.tex", "\n".join(lines))


def xai_table() -> None:
    x = pd.read_csv(WORK / "tables" / "xai_metrics.csv")
    x["b"] = x.model.map(lambda s: s.split("__")[0])
    x["nobox"] = x.model.str.contains("seg=0")
    order = [("resnet18", False), ("resnet50", False), ("efficientnet_b0", False), ("convnext_tiny", False), ("vit_small", False),
             ("swin_tiny", False), ("trace", True), ("trace", False)]
    lines = [r"\begin{table}[t]\centering\caption{Explanation grounding on the 493 D1 test images with tumour boxes (Grad-CAM of the "
             r"true class; 5 seeds). Pointing game: CAM maximum inside a box. Energy: share of positive CAM mass inside boxes. Stability: "
             r"correlation of CAM($x$) and the un-flipped CAM of the flipped image. $^{*}$ViT Grad-CAM is dominated by a few high-norm "
             r"background tokens and is not a reliable explanation for this architecture.}\label{tab:xai}",
             r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lccc}\toprule Model & Pointing & Energy & Stability \\ \midrule"]
    for b, nobox in order:
        r = x[(x.b == b) & (x.nobox == nobox)].iloc[0]
        name = MODEL_NAMES[b] + ("$^{*}$" if b == "vit_small" else "")
        if b == "trace":
            name += " w/o box supervision" if nobox else " (ours)"
            if nobox:
                lines.append(r"\midrule")
        lines.append(f"{name} & {pm(r.pointing_game_mean, r.pointing_game_std)} & {pm(r.energy_in_box_mean, r.energy_in_box_std)} & "
                     f"{pm(r.flip_stability_mean, r.flip_stability_std)} \\\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    write("tab_xai.tex", "\n".join(lines))
    t = x[(x.b == "trace") & ~x.nobox].iloc[0]
    n = x[(x.b == "trace") & x.nobox].iloc[0]
    cb = x[x.b == "convnext_tiny"].iloc[0]
    mac("PgTrace", f3(t.pointing_game_mean)); mac("PgTraceNoBox", f3(n.pointing_game_mean)); mac("PgTraceNoBoxSd", f3(n.pointing_game_std))
    mac("PgConvnext", f3(cb.pointing_game_mean)); mac("PgConvnextSd", f3(cb.pointing_game_std))
    mac("EnTrace", f3(t.energy_in_box_mean)); mac("EnConvnext", f3(cb.energy_in_box_mean)); mac("EnTraceNoBox", f3(n.energy_in_box_mean))
    mac("XaiImages", int(t.n_images_mean))
    c = pd.read_csv(WORK / "tables" / "all_central_runs.csv")
    nb = c[c.stage == "xai_control"]
    mac("NoBoxF", f3(nb.f1.mean()))


# =========================================================================== figures
def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif", "pdf.fonttype": 42})
    return plt


COLORS = {"trace": "#c0392b", "fedavg": "#2c3e50", "fedprox": "#2980b9", "scaffold": "#27ae60", "fedadam": "#8e44ad",
          "moon": "#d68910", "fedbn": "#7f8c8d", "fedper": "#95a5a6", "fedrep": "#bdc3c7"}


def fig_comm(f: pd.DataFrame) -> None:
    plt = _style()
    ff = f[(f.stage == "final") & (f.scenario == "main")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    markers = {"fedavg": "o", "fedprox": "s", "scaffold": "D", "fedadam": "^", "moon": "v", "fedbn": "<", "fedper": ">",
               "fedrep": "p", "trace": "*"}
    for ax, ds, title in zip(axes, ["dataset1", "dataset2"], ["D1 (4-class)", "D2 (binary)"]):
        g = ff[ff.dataset == ds].groupby("tag").agg(f1=("f1", "mean"), sd=("f1", "std"), up=("uplink_MB", "mean"), dn=("downlink_MB", "mean"))
        for i, t in enumerate(FL_ORDER):
            r = g.loc[t]
            # small horizontal jitter so methods with identical communication stay readable
            tot = (r.up + r.dn) * (1 + 0.06 * (i - 4) if t != "trace" else 1)
            ax.errorbar(tot, r.f1, yerr=r.sd, fmt=markers[t], ms=4 if t != "trace" else 11, color=COLORS[t], capsize=1.5,
                        lw=0.7, label=FL_NAMES[t].replace(" (ours)", ""))
        ax.set_xscale("log")
        ax.set_xlabel("total communication, up + down (MB, log scale)")
        ax.set_ylabel("test macro-F1")
        ax.set_title(title)
        ax.grid(alpha=0.3, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, frameon=False, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "f1_vs_communication.pdf")
    plt.close(fig)


def fig_convergence(f: pd.DataFrame) -> None:
    plt = _style()
    ff = f[(f.stage == "final") & (f.scenario == "main")].drop_duplicates("path")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, ds, title in zip(axes, ["dataset1", "dataset2"], ["D1 (4-class)", "D2 (binary)"]):
        for t in FL_ORDER:
            g = ff[(ff.dataset == ds) & (ff.tag == t)]
            curves = [[h["val_f1_clients"] for h in json.loads(Path(p).read_text())["history"]] for p in g.path]
            arr = np.array(curves)
            ax.plot(np.arange(1, arr.shape[1] + 1), arr.mean(0), color=COLORS[t], lw=1.8 if t == "trace" else 0.9,
                    label=FL_NAMES[t].replace(" (ours)", ""))
        ax.set_xlabel("communication round")
        ax.set_ylabel("pooled client-validation macro-F1")
        ax.set_title(title)
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=9, frameon=False, handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "convergence.pdf")
    plt.close(fig)


def _fig_sweep(g: pd.DataFrame) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    xs = [("dirichlet0.1_c5", "0.1"), ("dirichlet0.5_c5", "0.5"), ("dirichlet1.0_c5", "1.0"), ("iid_c5", "IID")]
    cs = [("dirichlet0.5_c3", "3"), ("dirichlet0.5_c5", "5"), ("dirichlet0.5_c10", "10")]
    for ax, sel, xl in [(axes[0], xs, r"Dirichlet $\alpha$ (5 clients)"), (axes[1], cs, r"clients ($\alpha=0.5$)")]:
        for m in ["fedavg", "fedprox", "scaffold", "trace"]:
            mu = [g.loc[(k, m), "mean"] for k, _ in sel]
            sd = [g.loc[(k, m), "std"] for k, _ in sel]
            ax.errorbar(range(len(sel)), mu, yerr=sd, color=COLORS[m], marker="*" if m == "trace" else "o",
                        ms=7 if m == "trace" else 4, capsize=2, lw=1.6 if m == "trace" else 0.9, label=FL_NAMES[m].replace(" (ours)", ""))
        ax.set_xticks(range(len(sel)), [lbl for _, lbl in sel])
        ax.set_xlabel(xl)
        ax.set_ylabel("D1 test macro-F1")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "heterogeneity.pdf")
    plt.close(fig)


def _fig_robust(r: pd.DataFrame) -> None:
    plt = _style()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for tag, label, col in [("trace", "TRACE-Fed (trust)", COLORS["trace"]), ("trace_no_trust", "sample-size weight", COLORS["fedavg"])]:
        curves = []
        for p in r[r.tag == tag].drop_duplicates("path").path:
            h = json.loads(Path(p).read_text())["history"]
            curves.append([[c["weight"] for c in x["clients"] if c["noisy"]][0] for x in h])
        arr = np.array(curves)
        ax.plot(np.arange(1, arr.shape[1] + 1), arr.mean(0), color=col, lw=1.5, label=label)
        if tag == "trace":  # seed range; the sample-size weight is constant within a seed
            ax.fill_between(np.arange(1, arr.shape[1] + 1), arr.min(0), arr.max(0), color=col, alpha=0.15, lw=0)
    ax.set_xlabel("communication round")
    ax.set_ylabel("weight of corrupted client")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "robust_weight.pdf")
    plt.close(fig)


def fig_xai() -> None:
    """Six test tumours: image+box, ConvNeXt-T, Swin-T, TRACE-Net w/o boxes, TRACE-Net CAM, TRACE-Net lesion map."""
    from .data import DataStore
    from .xai import analyse

    plt = _style()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = DataStore(Path("."), WORK, dev)
    ck = {
        "ConvNeXt-T": WORK / "checkpoints/final/dataset1/convnext_tiny__backbone_lr=0.0001/seed0.pt",
        "Swin-T": WORK / "checkpoints/final/dataset1/swin_tiny__backbone_lr=0.0001/seed0.pt",
        "TRACE-Net w/o boxes": WORK / "checkpoints/xai_control/dataset1/trace__backbone_lr=5e-05_seg=0_xc=0/seed0.pt",
        "TRACE-Net": WORK / "checkpoints/final/dataset1/trace__backbone_lr=5e-05/seed0.pt",
    }
    ex = {k: analyse(p, store)[1] for k, p in ck.items()}
    cols = 4
    rows = ["Image + box"] + list(ck) + ["TRACE-Net lesion map"]
    fig, axes = plt.subplots(len(rows), cols, figsize=(3.4, 3.65))
    first = ex["TRACE-Net"]
    for j in range(cols):
        for i, rname in enumerate(rows):
            ax = axes[i, j]
            ax.imshow(first["image"][j], cmap="gray")
            if rname in ck:
                cam = ex[rname]["cam"][j]
                ax.imshow(cam / (cam.max() + 1e-8), cmap="jet", alpha=0.45)
            elif rname == "TRACE-Net lesion map":
                ax.imshow(first["mask"][j], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax.contour(first["gt"][j], levels=[0.5], colors="lime", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rname, fontsize=6, rotation=0, ha="right", va="center")
    fig.subplots_adjust(left=0.30, right=0.99, top=0.99, bottom=0.01, wspace=0.03, hspace=0.03)
    fig.savefig(FIG / "xai_examples.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    c = load_central(WORK)
    f = load_fl(WORK)
    st = pd.read_csv(WORK / "tables" / "significance_paired_bootstrap.csv")
    data_audit()
    central_tables(c, st)
    fl_tables(f, st)
    ablation_tables(c, f)
    sweep_tables(f)
    cross_table(c)
    xai_table()
    fig_comm(f)
    fig_convergence(f)
    fig_xai()
    sel = json.loads((WORK / "selection.json").read_text())
    mac("SelTraceSpec", sel["central"]["trace"]["spec"].replace("_", r"\_"))
    write("numbers.tex", "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros, tables and figures written to {OUT}")


if __name__ == "__main__":
    main()
