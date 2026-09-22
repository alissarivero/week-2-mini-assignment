"""Texas last-statement analysis: apology vs religion by prior crime, plus themes.

Uses Texas Last Statement - CSV.csv (TDCJ executed-offender statements).
Optional Polars vs Pandas timing is at the bottom.
"""

from pathlib import Path
from time import perf_counter

import pandas as pd

from analysis import (
    DATA_PATH,
    FAMILY_EXACT,
    FAMILY_PREFIXES,
    GRATITUDE_LOVE_EXACT,
    GRATITUDE_LOVE_PREFIXES,
    RELIGION_EXACT,
    RELIGION_PREFIXES,
    REMORSE_EXACT,
    REMORSE_PREFIXES,
    add_text_scores,
    build_model_frame,
    coerce_numeric,
    fit_prior_models,
    fit_theme_models,
    label_prior_groups,
    load_statements,
    plot_prior_crime,
    plot_themes,
    prepare_demographic_frame,
)


def _repeat(fn, n=7, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    return sum(times) / len(times)


def _score_frame(frame):
    scored = add_text_scores(frame)
    scored["PreviousCrime"] = pd.to_numeric(scored["PreviousCrime"], errors="coerce")
    scored = scored.dropna(subset=["PreviousCrime"])
    scored["group"] = scored["PreviousCrime"].map({0.0: "No prior", 1.0: "Prior"})
    return scored.groupby("group")[
        ["apology_rate", "religion_rate", "gratitude_love_rate", "family_rate"]
    ].mean()


def pandas_pipeline():
    pdf = pd.read_csv(DATA_PATH, encoding="latin-1")
    pdf.columns = pdf.columns.str.strip()
    return _score_frame(pdf)


def polars_pipeline():
    import polars as pl

    pdf = pl.read_csv(
        str(DATA_PATH),
        encoding="utf8-lossy",
        null_values=["NA", ""],
        infer_schema_length=10000,
    )
    rename = {c: c.strip() for c in pdf.columns if c != c.strip()}
    if rename:
        pdf = pdf.rename(rename)

    remorse = "|".join(sorted(REMORSE_EXACT | set(REMORSE_PREFIXES)))
    gratitude = "|".join(sorted(GRATITUDE_LOVE_EXACT | set(GRATITUDE_LOVE_PREFIXES)))
    family = "|".join(sorted(FAMILY_EXACT | set(FAMILY_PREFIXES)))
    religion = "|".join(sorted(RELIGION_EXACT | set(RELIGION_PREFIXES)))
    declined_pat = r"(?i)^(none)?$|declined|no last statement"

    scored = (
        pdf.with_columns(
            pl.col("LastStatement").str.to_lowercase().alias("text"),
            pl.col("PreviousCrime").cast(pl.Float64, strict=False),
        )
        .with_columns(
            pl.col("text").str.count_matches(r"[A-Za-z']+").alias("word_count"),
            (
                pl.col("LastStatement").is_null()
                | pl.col("text").str.contains(declined_pat)
            ).alias("declined"),
            pl.col("text").str.count_matches(rf"(?i)\b(?:{remorse})").alias("apology_hits"),
            pl.col("text").str.count_matches(rf"(?i)\b(?:{gratitude})").alias("gratitude_love_hits"),
            pl.col("text").str.count_matches(rf"(?i)\b(?:{family})").alias("family_hits"),
            pl.col("text").str.count_matches(rf"(?i)\b(?:{religion})").alias("religion_hits"),
        )
        .with_columns(
            pl.when(pl.col("declined")).then(0).otherwise(pl.col("word_count")).alias("word_count")
        )
        .with_columns(
            pl.when(pl.col("word_count") == 0)
            .then(0.0)
            .otherwise(pl.col("apology_hits") / pl.col("word_count") * 100)
            .alias("apology_rate"),
            pl.when(pl.col("word_count") == 0)
            .then(0.0)
            .otherwise(pl.col("gratitude_love_hits") / pl.col("word_count") * 100)
            .alias("gratitude_love_rate"),
            pl.when(pl.col("word_count") == 0)
            .then(0.0)
            .otherwise(pl.col("family_hits") / pl.col("word_count") * 100)
            .alias("family_rate"),
            pl.when(pl.col("word_count") == 0)
            .then(0.0)
            .otherwise(pl.col("religion_hits") / pl.col("word_count") * 100)
            .alias("religion_rate"),
        )
        .filter(pl.col("PreviousCrime").is_in([0.0, 1.0]))
        .with_columns(
            pl.when(pl.col("PreviousCrime") == 0.0)
            .then(pl.lit("No prior"))
            .otherwise(pl.lit("Prior"))
            .alias("group")
        )
    )
    return scored.group_by("group").agg(
        pl.col("apology_rate").mean(),
        pl.col("religion_rate").mean(),
        pl.col("gratitude_love_rate").mean(),
        pl.col("family_rate").mean(),
    )


def main():
    df = load_statements()
    print("=== Load and inspect ===")
    print(df.head())
    print(df.info())
    print(df.describe())
    print("Missing values per column (empty or NA):")
    missing = df.apply(
        lambda col: col.isna().sum() + (col.astype(str).str.strip().isin(["", "NA"])).sum()
    )
    print(missing)

    df = coerce_numeric(df)
    df = add_text_scores(df)

    print(f"\nRows: {len(df)}")
    print(f"Declined / no statement: {int(df['declined'].sum())}")
    print("PreviousCrime counts (including NA):")
    print(df["PreviousCrime"].value_counts(dropna=False).sort_index())

    no_prior = df[df["PreviousCrime"] == 0]
    prior = df[df["PreviousCrime"] == 1]
    print(f"\nNo prior record: {len(no_prior)}")
    print(f"Prior record: {len(prior)}")

    print("\n=== groupby() by prior-crime group ===")
    labeled = label_prior_groups(df)
    print(
        labeled.groupby("group")[
            ["word_count", "Age", "EducationLevel", "apology_rate", "religion_rate"]
            + ["gratitude_love_rate", "family_rate"]
        ].agg(["mean", "count", "std"])
    )

    print("\n=== groupby() by Race ===")
    print(
        df.groupby("Race")[
            [
                "word_count",
                "Age",
                "EducationLevel",
                "remorse_rate",
                "gratitude_love_rate",
                "family_rate",
                "religion_rate",
            ]
        ].agg(["mean", "count", "std"])
    )

    comparison_metrics = [
        "word_count",
        "apology_rate",
        "religion_rate",
        "gratitude_love_rate",
        "family_rate",
    ]
    comparison = labeled.groupby(["Race", "group"])[comparison_metrics].mean().unstack()
    print("\n=== Side-by-side No prior vs Prior (means) ===")
    print(comparison)

    for metric in comparison_metrics:
        denom = comparison[(metric, "No prior")].replace(0, pd.NA)
        comparison[(metric, "percent_diff")] = (
            (comparison[(metric, "Prior")] - comparison[(metric, "No prior")]) / denom * 100
        )

    print("\n=== Percent difference (Prior minus No prior) / No prior ===")
    print(comparison.xs("percent_diff", axis=1, level=1))

    higher_apology = comparison[("apology_rate", "percent_diff")] > 0
    print("\nHigher apology rate with a prior record")
    print(comparison.loc[higher_apology, ("apology_rate", "percent_diff")])
    print("\nLower apology rate with a prior record")
    print(comparison.loc[~higher_apology, ("apology_rate", "percent_diff")])

    model_df = build_model_frame(labeled)
    print("\n=== Score means by group ===")
    print(
        model_df.groupby("prior_crime")[
            ["apology_rate", "religion_rate", "gratitude_love_rate", "family_rate"]
        ].agg(["mean", "count", "std"])
    )

    prior_models = fit_prior_models(model_df)
    print("\n=== apology_rate ~ prior_crime ===")
    print(prior_models["apology"].summary())
    print("\n=== religion_rate ~ prior_crime ===")
    print(prior_models["religion"].summary())

    demo = prepare_demographic_frame(model_df)
    theme_models = fit_theme_models(demo)
    for outcome, fitted in theme_models.items():
        print(f"\n=== {outcome} ~ prior_crime + Age + EducationLevel + Race ===")
        print(fitted.summary())

    out_plot = Path(__file__).resolve().parent / "apology_religion.png"
    plot_prior_crime(model_df, out_plot)
    print(f"\nSaved figure to {out_plot}")

    out_themes = Path(__file__).resolve().parent / "last_statement_themes.png"
    plot_themes(demo, out_themes)
    print(f"Saved figure to {out_themes}")

    from visuals import save_visuals

    gallery = save_visuals(df, labeled, demo, Path(__file__).resolve().parent)
    print("\nSaved gallery figures:")
    for name, path in gallery.items():
        print(f"  {name}: {path}")

    print("\n=== Pandas vs Polars (same pipeline on this CSV) ===")
    try:
        import polars as pl  # noqa: F401
    except ImportError:
        print("Polars is not installed. pip install polars")
    else:
        pandas_s = _repeat(pandas_pipeline)
        polars_s = _repeat(polars_pipeline)
        pandas_cmp = pandas_pipeline()
        polars_rows = {row["group"]: row for row in polars_pipeline().to_dicts()}
        diffs = []
        for group, row in pandas_cmp.iterrows():
            other = polars_rows.get(group)
            if other is None:
                continue
            for col in ("apology_rate", "religion_rate", "gratitude_love_rate", "family_rate"):
                if other.get(col) is None:
                    continue
                diffs.append(abs(float(row[col]) - float(other[col])))
        max_abs = max(diffs) if diffs else float("nan")
        print(f"Pandas mean time: {pandas_s * 1000:.1f} ms")
        print(f"Polars mean time: {polars_s * 1000:.1f} ms")
        print(f"Speedup (Pandas / Polars): {pandas_s / polars_s:.2f}x")
        print(f"Max abs difference on overlapping groups: {max_abs:.3e}")
        print(
            "Polars is compiled Rust under a Python API, so the same load / "
            "keyword-rate / group-by work is usually faster even on this "
            "small 545-row table."
        )


if __name__ == "__main__":
    main()
