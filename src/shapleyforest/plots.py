"""
Plotting utilities for ShapleyForestResult.

Python equivalents of all R shapff_plot.R functions:
  plot_importance()         — bar or beeswarm (shap summary)
  plot_waterfall()          — SHAP waterfall for one/multiple observations
  plot_force()              — SHAP force plot
  plot_stability()          — shadow-stability frequency bar chart
  plot_stability_elbow()    — per-pool stability curves with elbow
  plot_potential_interactions() — TreeSHAP interaction heatmap
  plot_decisions()          — cumulative SHAP decision plot
  plot_modules()            — module membership distribution
  plot_elbow()              — SHAP importance scree/elbow plot
  plot_soft_threshold()     — WGCNA soft-power diagnostic
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ── plot_importance ───────────────────────────────────────────────────────────

def plot_importance(
    result,
    kind: str = "beeswarm",
    max_display: int | None = None,
    color_bar_title: str = "Feature Value",
    fill: str = "#3e568a",
    sort_features: bool = True,
    show_numbers: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """
    Feature importance plot — bar or beeswarm.

    Matches R's plot_importance.shapley_forest().

    Parameters
    ----------
    kind : "bar", "beeswarm", or "both"
    max_display : max number of features shown (None = all)
    fill : colour for the bar chart
    """
    shap_values = result.shap_obj["shapley_values"]
    X           = result.final_X
    feat_names  = list(X.columns)

    if max_display is None:
        max_display = len(feat_names)

    if kind == "bar":
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order    = np.argsort(mean_abs)[::-1][:max_display]
        names    = [feat_names[i] for i in order]
        values   = mean_abs[order]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, max_display * 0.35)))

        bars = ax.barh(range(len(names)), values[::-1], color=fill)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Feature Importance", fontsize=14, ha="center")
        if show_numbers:
            for i, (bar, val) in enumerate(zip(bars, values[::-1])):
                ax.text(bar.get_width() + 0.001 * values.max(),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=8)
        ax.invert_yaxis()
        plt.tight_layout()
        return ax

    elif kind == "beeswarm":
        import shap as _shap
        _shap.summary_plot(
            shap_values,
            X,
            max_display=max_display,
            show=False,
            sort=sort_features,
            color_bar_label=color_bar_title,
        )
        plt.title("Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.show()
        return None

    elif kind == "both":
        import shap as _shap
        fig, axes = plt.subplots(1, 2, figsize=(16, max(5, max_display * 0.4)))

        # bar
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        order    = np.argsort(mean_abs)[::-1][:max_display]
        names    = [feat_names[i] for i in order]
        values   = mean_abs[order]
        axes[0].barh(range(len(names)), values[::-1], color=fill)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names[::-1])
        axes[0].set_xlabel("Mean |SHAP value|")
        axes[0].set_title("Bar", fontsize=13)
        axes[0].invert_yaxis()

        # beeswarm via shap
        plt.sca(axes[1])
        _shap.summary_plot(
            shap_values, X, max_display=max_display,
            show=False, sort=sort_features,
            color_bar_label=color_bar_title,
        )
        axes[1].set_title("Beeswarm", fontsize=13)

        fig.suptitle("Feature Importance", fontsize=14)
        plt.tight_layout()
        plt.show()
        return None
    else:
        raise ValueError("kind must be 'bar', 'beeswarm', or 'both'")


# ── plot_waterfall ────────────────────────────────────────────────────────────

def plot_waterfall(
    result,
    row_id: int | list,
    row_name: str | list | None = None,
    max_display: int | None = None,
) -> None:
    """
    SHAP waterfall plot for one or more observations.

    Matches R's plot_waterfall.shapley_forest().
    When multiple row_ids are given, SHAP values are averaged.
    """
    import shap as _shap

    shap_values = result.shap_obj["shapley_values"]
    X           = result.final_X.values

    if isinstance(row_id, (int, np.integer)):
        row_ids = [int(row_id)]
    else:
        row_ids = [int(r) for r in row_id]

    sv = shap_values[row_ids].mean(axis=0) if len(row_ids) > 1 else shap_values[row_ids[0]]
    x  = X[row_ids].mean(axis=0)           if len(row_ids) > 1 else X[row_ids[0]]

    if row_name is None:
        label = (f"Samples {', '.join(str(r) for r in row_ids)}"
                 if len(row_ids) > 1 else f"Sample {row_ids[0]}")
    else:
        rn = row_name if isinstance(row_name, list) else [row_name]
        label = (f"Samples {', '.join(str(r) for r in rn)}"
                 if len(rn) > 1 else f"Sample {rn[0]}")

    n_display = max_display or len(result.final_X.columns)
    base_val  = float(np.mean(result.final_rf.predict(result.final_X.values)))

    expl = _shap.Explanation(
        values         = sv,
        base_values    = base_val,
        data           = x,
        feature_names  = list(result.final_X.columns),
    )
    _shap.plots.waterfall(expl, max_display=n_display, show=False)
    plt.title(f"Waterfall Plot — {label}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── plot_force ────────────────────────────────────────────────────────────────

def plot_force(
    result,
    row_id: int | list,
    row_name: str | list | None = None,
    max_display: int | None = None,
) -> None:
    """
    SHAP force plot for one or more observations.

    Matches R's plot_force.shapley_forest().
    """
    import shap as _shap

    shap_values = result.shap_obj["shapley_values"]
    X           = result.final_X.values

    if isinstance(row_id, (int, np.integer)):
        row_ids = [int(row_id)]
    else:
        row_ids = [int(r) for r in row_id]

    sv = shap_values[row_ids].mean(axis=0) if len(row_ids) > 1 else shap_values[row_ids[0]]
    x  = X[row_ids].mean(axis=0)           if len(row_ids) > 1 else X[row_ids[0]]

    base_val = float(np.mean(result.final_rf.predict(result.final_X.values)))

    if row_name is None:
        label = (f"Samples {', '.join(str(r) for r in row_ids)}"
                 if len(row_ids) > 1 else f"Sample {row_ids[0]}")
    else:
        rn = row_name if isinstance(row_name, list) else [row_name]
        label = (f"Samples {', '.join(str(r) for r in rn)}"
                 if len(rn) > 1 else f"Sample {rn[0]}")

    _shap.force_plot(
        base_val, sv, x,
        feature_names  = list(result.final_X.columns),
        matplotlib     = True,
        show           = False,
    )
    plt.title(f"Force Plot — {label}", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── plot_stability ────────────────────────────────────────────────────────────

def plot_stability(
    result,
    fill: str = "#3e568a",
    main: str = "Shadow-Stability Frequencies",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Horizontal bar chart of shadow-stability selection frequencies.

    Matches R's plot_stability.shapley_forest().
    """
    freq_dict = result.stability_freq
    if not freq_dict:
        raise ValueError("No stability frequencies in this result.")

    df = (pd.DataFrame({"feature": list(freq_dict.keys()),
                         "frequency": list(freq_dict.values())})
            .sort_values("frequency"))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))

    ax.barh(range(len(df)), df["frequency"], color=fill)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Selection frequency")
    ax.set_title(main, fontsize=14)
    plt.tight_layout()
    return ax


# ── plot_stability_elbow ──────────────────────────────────────────────────────

def plot_stability_elbow(
    result,
    col_selected: str   = "#3e568a",
    col_unselected: str = "gray",
    col_threshold: str  = "tomato",
    col_elbow: str      = "#e07b00",
    show_labels: bool   = False,
    ncol: int | None    = None,
    main: str           = "Shadow-stability selection",
) -> plt.Figure:
    """
    Per-pool stability frequency curves with threshold and elbow lines.

    Matches R's plot_stability_elbow.shapley_forest().
    """
    sd = result.stability_data
    if sd is None or len(sd) == 0:
        raise ValueError(
            "No stability selection data in this result. "
            "Refit with sf() to populate stability_data."
        )

    pools = sd["pool"].unique()
    n_pools = len(pools)
    ncols   = ncol or min(n_pools, 3)
    nrows   = int(np.ceil(n_pools / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)

    for ax_idx, pool in enumerate(pools):
        r, c = divmod(ax_idx, ncols)
        ax   = axes[r][c]
        pdata = sd[sd["pool"] == pool].copy()

        colours = [col_selected if s else col_unselected
                   for s in pdata["selected"]]
        ax.plot(pdata["rank"], pdata["freq"], color="lightgray", linewidth=0.7)
        ax.scatter(pdata["rank"], pdata["freq"], c=colours, s=30, alpha=0.9, zorder=3)

        thr = pdata["threshold"].iloc[0]
        ax.axhline(thr, color=col_threshold, linestyle="--", linewidth=0.9,
                   label=f"threshold={thr:.2f}")

        elbow_r = pdata["elbow_rank"].iloc[0]
        if pd.notna(elbow_r) and elbow_r is not None:
            elbow_r = int(elbow_r)
            ax.axvline(elbow_r, color=col_elbow, linestyle=":", linewidth=0.9)
            ax.text(elbow_r, 0.97, f"elbow\nrank {elbow_r}",
                    color=col_elbow, fontsize=7, ha="left", va="top")

        if show_labels:
            for _, row in pdata.iterrows():
                ax.text(row["rank"], row["freq"], row["feature"],
                        fontsize=6, ha="left", va="bottom")

        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlabel("Feature rank (freq desc)")
        ax.set_ylabel("Stability frequency")
        ax.set_title(pool, fontweight="bold", fontsize=11)
        ax.legend(fontsize=7)

    # hide empty subplots
    for ax_idx in range(n_pools, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(main, fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


# ── plot_potential_interactions ───────────────────────────────────────────────

def plot_potential_interactions(result) -> plt.Figure:
    """
    Heatmap of mean absolute TreeSHAP pairwise interaction strengths.

    Matches R's plot_potential_interactions.shapley_forest().
    """
    int_mat = result.interaction_matrix   # lazy: computes + caches on first access
    if int_mat is None:
        raise ValueError(
            "No interaction matrix. Refit with compute_interactions=True."
        )

    feats = list(result.final_X.columns)
    p     = len(feats)

    # build full symmetric matrix as DataFrame
    df_full = pd.DataFrame(int_mat, index=feats, columns=feats)

    fig, ax = plt.subplots(figsize=(max(6, p * 0.7), max(5, p * 0.65)))
    mid_val = float(np.median(int_mat[np.triu_indices(p, k=1)]))

    cmap  = plt.get_cmap("RdBu_r")
    vmax  = float(int_mat.max())
    vmin  = 0.0
    im    = ax.imshow(int_mat, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(feats, fontsize=9)

    for i in range(p):
        for j in range(p):
            ax.text(j, i, f"{int_mat[i, j]:.3f}",
                    ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Mean |SHAP interaction|")
    ax.set_title("TreeSHAP Interaction Values", fontsize=14)
    ax.set_xlabel(None)
    fig.text(0.5, -0.01,
             "Off-diagonal: pairwise interactions  |  Diagonal: main effects",
             ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()
    return fig


# ── plot_decisions ────────────────────────────────────────────────────────────

def plot_decisions(
    result,
    highlight: list | None = None,
    just_shap: bool = False,
    plot_title: str = "Decision Plot",
    geom_point: bool = False,
    gradient: tuple = ("blue", "red"),
) -> plt.Figure:
    """
    Cumulative SHAP decision plot.

    Matches R's plot_decisions.shapley_forest().
    """
    if just_shap:
        if not isinstance(result, (np.ndarray, pd.DataFrame)):
            raise ValueError("When just_shap=True, pass a SHAP array/DataFrame directly.")
        shap_values = np.asarray(result)
        feat_names  = [f"F{i}" for i in range(shap_values.shape[1])]
        base_val    = None
    else:
        shap_values = result.shap_obj["shapley_values"]
        feat_names  = list(result.final_X.columns)
        base_val    = float(np.mean(result.final_rf.predict(result.final_X.values)))

    # sort features by mean |SHAP| ascending (bottom of plot = least important)
    mean_abs   = np.mean(np.abs(shap_values), axis=0)
    feat_order = np.argsort(mean_abs)                    # ascending
    feat_sorted = [feat_names[i] for i in feat_order]
    sv_sorted  = shap_values[:, feat_order]

    n_obs = sv_sorted.shape[0]
    if highlight is not None:
        obs_idx = [int(h) for h in highlight]
    else:
        obs_idx = list(range(n_obs))

    sv_sub = sv_sorted[obs_idx]
    cumsum = np.cumsum(sv_sub, axis=1)                   # (n_sub, p)

    # terminal value for colour scaling
    last_val = cumsum[:, -1]
    norm     = mcolors.Normalize(vmin=last_val.min(), vmax=last_val.max())
    cmap     = mcolors.LinearSegmentedColormap.from_list("grad", gradient)

    # prepend baseline column of zeros
    full_cum = np.hstack([np.zeros((len(obs_idx), 1)), cumsum])
    y_ticks  = range(len(feat_sorted) + 1)

    fig, ax = plt.subplots(figsize=(8, max(5, len(feat_sorted) * 0.4)))

    for i, obs in enumerate(obs_idx):
        colour = cmap(norm(last_val[i]))
        ax.plot(full_cum[i], y_ticks, color=colour, linewidth=0.8, alpha=0.7)
        if geom_point:
            ax.scatter(full_cum[i], y_ticks, color=colour, s=10, zorder=3)

    if base_val is not None:
        ax.axvline(base_val, color="#999999", linestyle="--", linewidth=1)

    ax.set_yticks(list(y_ticks))
    ax.set_yticklabels([""] + feat_sorted, fontsize=9)
    ax.set_xlabel("Cumulative SHAP Value", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title(plot_title, fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Predicted Output")
    plt.tight_layout()
    plt.show()
    return fig


# ── plot_modules ──────────────────────────────────────────────────────────────

def plot_modules(
    result,
    main: str  = "Module Membership Distribution",
    xlab: str  = "Module",
    ylab: str  = "Proportion of features",
    module_labels: pd.DataFrame | None = None,
) -> plt.Figure:
    """
    Stacked bar chart of selected vs not-selected proportion per module.

    Matches R's plot_modules.shapley_forest().
    """
    mm = result.module_membership.copy()   # feature_name, module

    if module_labels is not None:
        lmap = dict(zip(module_labels.iloc[:, 0], module_labels.iloc[:, 1]))
        mm["module"] = mm["module"].map(lmap).fillna(mm["module"])

    all_mods   = mm["module"].value_counts()
    selected   = set(result.final_X.columns)
    sel_mods   = mm[mm["feature_name"].isin(selected)]["module"].value_counts()

    total = len(mm)
    mods  = all_mods.index.tolist()

    sel_pct   = [(sel_mods.get(m, 0) / total) for m in mods]
    unsel_pct = [(all_mods[m] - sel_mods.get(m, 0)) / total for m in mods]

    fig, ax = plt.subplots(figsize=(max(6, len(mods) * 0.8), 5))
    x = range(len(mods))
    ax.bar(x, unsel_pct, label="Not selected", color="#cccccc")
    ax.bar(x, sel_pct,   label="Selected",     color="#3e568a", bottom=unsel_pct)
    ax.set_xticks(list(x))
    ax.set_xticklabels(mods, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(main, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


# ── plot_elbow ────────────────────────────────────────────────────────────────

def plot_elbow(
    result,
    main: str      = "Elbow Plot of Absolute SHAP Values",
    xlab: str      = "Features",
    ylab: str      = "Mean |SHAP value|",
    color: bool    = True,
    color_name: str = "Module",
    module_labels: pd.DataFrame | None = None,
) -> plt.Figure:
    """
    Scree/elbow plot of mean |SHAP| by stable feature rank.

    Matches R's plot_elbow.shapley_forest().
    """
    df = result.final_SHAP.copy()

    if module_labels is not None:
        lmap = dict(zip(module_labels.iloc[:, 0], module_labels.iloc[:, 1]))
        df["module_membership"] = df["module_membership"].map(lmap).fillna(
            df["module_membership"])

    df = df.reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))

    if color and "module_membership" in df.columns:
        unique_mods = df["module_membership"].unique()
        cmap_m      = plt.get_cmap("tab20", len(unique_mods))
        mod_colour  = {m: cmap_m(i) for i, m in enumerate(unique_mods)}
        colours     = df["module_membership"].map(mod_colour)
    else:
        colours = "#3e568a"

    ax.plot(df["rank"], df["variable_importance"], color="gray", linewidth=1, zorder=1)
    ax.scatter(df["rank"], df["variable_importance"], c=colours, s=60, zorder=2)

    ax.set_xticks(df["rank"])
    ax.set_xticklabels(df["feature_name"], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(main, fontsize=14)

    if color and "module_membership" in df.columns:
        handles = [plt.scatter([], [], c=[mod_colour[m]], label=m, s=40)
                   for m in unique_mods]
        ax.legend(handles=handles, title=color_name, fontsize=8)

    plt.tight_layout()
    plt.show()
    return fig


# ── plot_signed_importance ────────────────────────────────────────────────────

def plot_signed_importance(
    result,
    top_n: int | None = None,
    pos_color: str = "#d6604d",
    neg_color: str = "#4393c3",
    ambiguous_color: str = "#bbbbbb",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Diverging bar chart of signed importance = direction × mean|SHAP|.

    Magnitude is mean|SHAP|; sign is the global effect direction (from the
    Spearman correlation between each feature's value and its SHAP value).
    Bars to the right promote the outcome, to the left suppress it. Features
    whose direction is ambiguous (non-monotone: |Spearman| < 0.1) are greyed
    out — for those a single direction is misleading; use a SHAP dependence
    plot instead.
    """
    df = result.final_SHAP.copy()
    if "signed_importance" not in df.columns or len(df) == 0:
        raise ValueError("No directional SHAP data in this result (no stable features).")

    df = df.reindex(df["signed_importance"].abs().sort_values().index)
    if top_n is not None:
        df = df.iloc[-top_n:]

    colours = []
    for _, row in df.iterrows():
        if row.get("direction_ambiguous", False):
            colours.append(ambiguous_color)
        elif row["signed_importance"] >= 0:
            colours.append(pos_color)
        else:
            colours.append(neg_color)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))

    ax.barh(range(len(df)), df["signed_importance"], color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature_name"])
    ax.set_xlabel("Signed importance  (direction × mean |SHAP|)")
    ax.set_title("Directional feature importance", fontsize=14)

    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=pos_color, label="promotes outcome (+)"),
        Patch(color=neg_color, label="suppresses outcome (−)"),
        Patch(color=ambiguous_color, label="ambiguous (non-monotone)"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()
    return ax


# ── plot_soft_threshold ───────────────────────────────────────────────────────

def plot_soft_threshold(
    datout: pd.DataFrame,
    chosen_power: int,
    r2_threshold: float = 0.85,
) -> plt.Figure:
    """
    Two-panel WGCNA soft-threshold diagnostic.

    Matches R's scale-free topology + mean connectivity plot.
    datout must have columns "Power", "SFT.R.sq", "mean.k."
    """
    powers  = datout["Power"].tolist()
    r2      = datout["SFT.R.sq"].tolist()
    mean_k  = datout["mean.k."].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(powers, r2, "o-", color="#4C72B0")
    ax1.axhline(r2_threshold, color="red", linestyle="--", linewidth=1,
                label=f"R² = {r2_threshold}")
    ax1.axvline(chosen_power, color="grey", linestyle=":", linewidth=1,
                label=f"Chosen β = {chosen_power}")
    ax1.set_xlabel("Soft-threshold power (β)")
    ax1.set_ylabel("Scale-free topology fit (R²)")
    ax1.set_title("Scale-free topology fit")
    ax1.legend(fontsize=9)

    ax2.plot(powers, mean_k, "o-", color="#DD8452")
    ax2.axvline(chosen_power, color="grey", linestyle=":", linewidth=1,
                label=f"Chosen β = {chosen_power}")
    ax2.set_xlabel("Soft-threshold power (β)")
    ax2.set_ylabel("Mean connectivity")
    ax2.set_title("Mean connectivity")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig
