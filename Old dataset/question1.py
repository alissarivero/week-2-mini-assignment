"""ASD gut microbiome analysis: good vs bad bacteria in ASD vs controls.

Uses ASD meta abundance.csv (Dan et al., 2020). Optional Polars vs Pandas
timing is at the bottom.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

DATA_PATH = Path(__file__).resolve().parent / "ASD meta abundance.csv"

# Load and inspect: shape, dtypes, summary stats, and missing values.
df = pd.read_csv(DATA_PATH)
print("=== Load and inspect ===")
print(df.head())
print(df.info())
print(df.describe())
print("Missing values per column:")
print(df.isnull().sum())

# A* columns are ASD samples; B* columns are typically developing controls.
autism_cols = [c for c in df.columns if c.startswith("A")]
control_cols = [c for c in df.columns if c.startswith("B")]
print(f"\nASD samples: {len(autism_cols)}")
print(f"Control samples: {len(control_cols)}")

# Long format so groupby() can summarize by group (ASD vs Control) and by genus.
long_df = df.melt(
    id_vars="Taxonomy",
    var_name="sample",
    value_name="abundance",
)
long_df["group"] = long_df["sample"].map(
    lambda s: "ASD" if s.startswith("A") else "Control"
)
long_df["genus"] = long_df["Taxonomy"].str.extract(r"g__([^;]+)")

print("\n=== groupby() by group ===")
print(long_df.groupby("group")["abundance"].agg(["mean", "count", "std"]))

print("\n=== groupby() by genus (top 15 by mean) ===")
genus_stats = (
    long_df.groupby("genus")["abundance"]
    .agg(["mean", "count", "sum"])
    .sort_values("mean", ascending=False)
)
print(genus_stats.head(15))

# Relative abundance: each sample's counts divided by that sample's total reads.
rel = df.set_index("Taxonomy")
rel = rel.div(rel.sum(axis=0), axis=1)

long_rel = rel.reset_index().melt(
    id_vars="Taxonomy",
    var_name="sample",
    value_name="rel_abundance",
)
long_rel["group"] = long_rel["sample"].map(
    lambda s: "ASD" if s.startswith("A") else "Control"
)
long_rel["genus"] = long_rel["Taxonomy"].str.extract(r"g__([^;]+)")

comparison = long_rel.groupby(["genus", "group"])["rel_abundance"].mean().unstack()
comparison["percent_diff"] = (
    (comparison["ASD"] - comparison["Control"]) / comparison["Control"] * 100
)
comparison = comparison.replace([float("inf"), float("-inf")], pd.NA)
comparison = comparison.dropna(subset=["percent_diff"])
comparison["abs_percent_diff"] = comparison["percent_diff"].abs()
comparison = comparison.sort_values("abs_percent_diff", ascending=False)

comparison_out = comparison.copy()
comparison_out["ASD"] = comparison_out["ASD"] * 100
comparison_out["Control"] = comparison_out["Control"] * 100
cols = ["ASD", "Control", "percent_diff", "abs_percent_diff"]

print("\n=== Side-by-side ASD vs Control (percent of reads) ===")
print(comparison_out[cols].head(30))

# Keep genera that reach at least 0.01% in ASD or Control, then split by sign of the difference.
filtered = comparison_out[
    (comparison_out["ASD"] > 0.01) | (comparison_out["Control"] > 0.01)
]
print("\n=== Genera with ASD or Control > 0.01% ===")
print(filtered[cols].head(30))

higher_in_asd = filtered[filtered["percent_diff"] > 0]
lower_in_asd = filtered[filtered["percent_diff"] < 0]
print("\nHigher in ASD")
print(higher_in_asd[cols].head(30))
print("\nLower in ASD")
print(lower_in_asd[cols].head(30))

# Paper-based beneficial vs harmful taxa. A taxon listed as harmful wins if both apply.
GOOD_GENERA = {
    "Bifidobacterium",
    "Lactobacillus",
    "Faecalibacterium",
    "Roseburia",
    "Ruminococcus",
    "Bacteroides",
    "Prevotella",
}
BAD_GENERA = {
    "Salmonella",
    "Shigella",
    "Klebsiella",
    "Serratia",
    "Pseudomonas",
    "Campylobacter",
    "Desulfovibrio",
    "Enterobacter",
    "Citrobacter",
    "Proteus",
    "Yersinia",
    "Fusobacterium",
    "Leptotrichia",
    "Veillonella",
    "Megamonas",
    "Megasphaera",
    "Dialister",
    "Allisonella",
}


def taxon_genus(tax):
    if tax.startswith("g__"):
        return tax.split(";")[0].removeprefix("g__")
    return ""


def is_gnavus(tax):
    return "gnavus" in tax.lower()


def is_bad_taxon(tax):
    genus = taxon_genus(tax)
    if is_gnavus(tax):
        return True
    if "Clostridioides difficile" in tax:
        return True
    if tax.endswith("s__Escherichia coli"):
        return True
    return genus in BAD_GENERA


def is_good_taxon(tax):
    if is_bad_taxon(tax):
        return False
    genus = taxon_genus(tax)
    if genus in GOOD_GENERA:
        return True
    return "Oxalobacter formigenes" in tax


good_taxa = df.loc[df["Taxonomy"].map(is_good_taxon), "Taxonomy"]
bad_taxa = df.loc[df["Taxonomy"].map(is_bad_taxon), "Taxonomy"]
print(f"\nn_good_taxa: {len(good_taxa)}")
print(f"n_bad_taxa: {len(bad_taxa)}")
print("Unmatched paper names: Helicobacter hepaticus")

sample_df = pd.DataFrame(
    {
        "sample": rel.columns,
        "autism": [1 if c.startswith("A") else 0 for c in rel.columns],
        "good_bacteria": rel.loc[good_taxa].sum(axis=0).values,
        "bad_bacteria": rel.loc[bad_taxa].sum(axis=0).values,
    }
)
print("\n=== Score means by group ===")
print(
    sample_df.groupby("autism")[["good_bacteria", "bad_bacteria"]].agg(
        ["mean", "count", "std"]
    )
)

# OLS: autism (1 = ASD, 0 = control) predicting each composite score.
X = sm.add_constant(sample_df["autism"])
good_model = sm.OLS(sample_df["good_bacteria"], X).fit()
bad_model = sm.OLS(sample_df["bad_bacteria"], X).fit()
print("\n=== good_bacteria ~ autism ===")
print(good_model.summary())
print("\n=== bad_bacteria ~ autism ===")
print(bad_model.summary())

# Boxplot + strip: two groups, continuous scores, n = 30 each. The box shows
# the distribution; points keep every sample visible. Separate y-axes because
# the bad-bacteria score is a much smaller share of reads.
sns.set_theme(style="ticks", context="notebook")
plot_df = sample_df.copy()
plot_df["group"] = plot_df["autism"].map({0: "Control", 1: "ASD"})
plot_df["Good bacteria"] = plot_df["good_bacteria"] * 100
plot_df["Bad bacteria"] = plot_df["bad_bacteria"] * 100

panels = [
    ("Good bacteria", {"Control": "#D7E2B4", "ASD": "#5E7A32"}, "#3F5320"),
    ("Bad bacteria", {"Control": "#F3D6E0", "ASD": "#B56B86"}, "#7A3F56"),
]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), facecolor="#FBF9F6")
fig.patch.set_facecolor("#FBF9F6")
for ax, (col, palette, edge) in zip(axes, panels):
    ax.set_facecolor("#FBF9F6")
    sns.boxplot(
        data=plot_df,
        x="group",
        y=col,
        hue="group",
        hue_order=["Control", "ASD"],
        order=["Control", "ASD"],
        palette=palette,
        width=0.55,
        linewidth=1.1,
        fliersize=0,
        legend=False,
        ax=ax,
    )
    for patch in ax.patches:
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.1)
        patch.set_alpha(0.95)
    sns.stripplot(
        data=plot_df,
        x="group",
        y=col,
        hue="group",
        hue_order=["Control", "ASD"],
        order=["Control", "ASD"],
        palette=palette,
        legend=False,
        jitter=0.18,
        size=5.5,
        alpha=0.7,
        linewidth=0.6,
        edgecolor="white",
        ax=ax,
    )
    ax.set_title(col, pad=10, color=edge)
    ax.set_xlabel("")
    ax.set_ylabel("Share of reads (%)")
    sns.despine(ax=ax, trim=True)
    ax.spines["left"].set_color("#C9C3B8")
    ax.spines["bottom"].set_color("#C9C3B8")
    ax.yaxis.grid(True, color="#E6E1D8", linewidth=0.8)
    ax.set_axisbelow(True)

fig.suptitle(
    "Gut bacteria scores in control vs ASD children",
    y=1.03,
    fontsize=16,
    fontweight="semibold",
    color="#2F2B26",
)
fig.tight_layout()
out_plot = Path(__file__).resolve().parent / "good_bad_bacteria.png"
fig.savefig(out_plot, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved figure to {out_plot}")


def _repeat(fn, n=7, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    return sum(times) / len(times)


def pandas_pipeline():
    pdf = pd.read_csv(DATA_PATH)
    sample_cols = [c for c in pdf.columns if c != "Taxonomy"]
    totals = pdf[sample_cols].sum(axis=0)
    rel_pdf = pdf[sample_cols].div(totals, axis=1)
    long = rel_pdf.assign(Taxonomy=pdf["Taxonomy"]).melt(
        id_vars="Taxonomy", var_name="sample", value_name="rel_abundance"
    )
    long["group"] = long["sample"].map(
        lambda s: "ASD" if s.startswith("A") else "Control"
    )
    long["genus"] = long["Taxonomy"].str.extract(r"g__([^;]+)")
    return long.groupby(["genus", "group"])["rel_abundance"].mean().unstack()


def polars_pipeline():
    import polars as pl

    pdf = pl.read_csv(DATA_PATH)
    sample_cols = [c for c in pdf.columns if c != "Taxonomy"]
    totals = pdf.select([pl.col(c).sum().alias(c) for c in sample_cols])
    rel_pdf = pdf.select(
        ["Taxonomy"]
        + [(pl.col(c) / totals.get_column(c)[0]).alias(c) for c in sample_cols]
    )
    long = rel_pdf.unpivot(
        index="Taxonomy", variable_name="sample", value_name="rel_abundance"
    ).with_columns(
        pl.when(pl.col("sample").str.starts_with("A"))
        .then(pl.lit("ASD"))
        .otherwise(pl.lit("Control"))
        .alias("group"),
        pl.col("Taxonomy").str.extract(r"g__([^;]+)", 1).alias("genus"),
    )
    return long.group_by(["genus", "group"]).agg(pl.col("rel_abundance").mean()).pivot(
        values="rel_abundance", index="genus", on="group"
    )


# Same load / relative-abundance / genus-mean pipeline in Pandas and Polars.
print("\n=== Pandas vs Polars (same pipeline on this CSV) ===")
try:
    import polars as pl  # noqa: F401
except ImportError:
    print("Polars is not installed. pip install polars")
else:
    pandas_s = _repeat(pandas_pipeline)
    polars_s = _repeat(polars_pipeline)
    pandas_cmp = pandas_pipeline()
    polars_rows = {row["genus"]: row for row in polars_pipeline().to_dicts()}
    diffs = []
    for genus, row in pandas_cmp.iterrows():
        other = polars_rows.get(genus)
        if other is None:
            continue
        for col in ("ASD", "Control"):
            if other.get(col) is None:
                continue
            diffs.append(abs(float(row[col]) - float(other[col])))
    max_abs = max(diffs) if diffs else float("nan")
    print(f"Pandas mean time: {pandas_s * 1000:.1f} ms")
    print(f"Polars mean time: {polars_s * 1000:.1f} ms")
    print(f"Speedup (Pandas / Polars): {pandas_s / polars_s:.2f}x")
    print(f"Max abs difference on overlapping genera: {max_abs:.3e}")
    print(
        "Polars is compiled Rust under a Python API, so the same load / "
        "relative-abundance / group-by work is usually faster even on this "
        "small 5,619 x 60 table."
    )
