"""Importable Texas last-statement analysis functions.

question1.py is the script that prints the full report. Tests import this module.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

DATA_PATH = Path(__file__).resolve().parent / "Texas Last Statement - CSV.csv"

NUMERIC_COLS = (
    "Age",
    "AgeWhenReceived",
    "EducationLevel",
    "PreviousCrime",
    "Codefendants",
    "NumberVictim",
)

REMORSE_EXACT = {
    "sorry",
    "sorrow",
    "apology",
    "apologize",
    "apologise",
    "apologized",
    "apologised",
    "apologies",
    "forgive",
    "forgave",
    "forgiven",
    "forgiveness",
    "forgiving",
    "remorse",
    "remorseful",
    "regret",
    "regrets",
    "regretted",
    "regretting",
    "repent",
    "repents",
    "repented",
    "repentance",
}
REMORSE_PREFIXES = ("apolog", "forgiv", "remorse", "regret", "repent")

GRATITUDE_LOVE_EXACT = {
    "thank",
    "thanks",
    "thankful",
    "thankfully",
    "gratitude",
    "grateful",
    "appreciate",
    "appreciated",
    "appreciation",
    "love",
    "loved",
    "loves",
    "loving",
    "lovin",
}
GRATITUDE_LOVE_PREFIXES = ("thank", "gratitude", "grateful", "appreciate")

FAMILY_EXACT = {
    "family",
    "families",
    "mom",
    "moms",
    "mommy",
    "mama",
    "mother",
    "mothers",
    "mum",
    "dad",
    "dads",
    "daddy",
    "father",
    "fathers",
    "papa",
    "parent",
    "parents",
    "wife",
    "wives",
    "husband",
    "husbands",
    "son",
    "sons",
    "daughter",
    "daughters",
    "kid",
    "kids",
    "child",
    "children",
    "brother",
    "brothers",
    "bro",
    "sister",
    "sisters",
    "sis",
    "sibling",
    "siblings",
    "grandma",
    "grandmother",
    "grandpa",
    "grandfather",
    "grandparent",
    "grandparents",
    "aunt",
    "aunts",
    "uncle",
    "uncles",
    "nephew",
    "nephews",
    "niece",
    "nieces",
    "cousin",
    "cousins",
    "friend",
    "friends",
    "friendship",
    "baby",
    "babies",
}
FAMILY_PREFIXES = (
    "mother",
    "father",
    "daughter",
    "brother",
    "sister",
    "grandma",
    "grandpa",
    "grandparent",
    "friend",
)

RELIGION_EXACT = {
    "god",
    "gods",
    "godly",
    "jesus",
    "christ",
    "christian",
    "christianity",
    "lord",
    "lords",
    "heaven",
    "heavenly",
    "allah",
    "pray",
    "prayer",
    "prayers",
    "praying",
    "prayed",
    "bible",
    "biblical",
    "amen",
    "holy",
    "church",
    "faith",
    "bless",
    "blessed",
    "blessing",
    "blessings",
}
RELIGION_PREFIXES = ("jesus", "christ", "heaven", "pray", "prayer", "bible", "bless")

THEME_RATES = [
    "apology_rate",
    "remorse_rate",
    "gratitude_love_rate",
    "family_rate",
    "religion_rate",
]


def tokenize(text):
    words = []
    current = []
    for ch in str(text).lower():
        if ch.isalpha() or ch == "'":
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words


def _is_hit(word, exact, prefixes):
    if word in exact:
        return True
    return any(word.startswith(prefix) for prefix in prefixes)


def theme_hits(words, exact, prefixes):
    return sum(_is_hit(word, exact, prefixes) for word in words)


def is_declined(text):
    if pd.isna(text):
        return True
    low = str(text).strip().lower()
    return low in {"none", "", "nan"} or "declined" in low or "no last statement" in low


def load_statements(path=None):
    """Load the TDCJ CSV (Latin-1) and strip header whitespace."""
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Statement file not found: {path}")
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    return df


def coerce_numeric(df, columns=NUMERIC_COLS):
    """Turn Age / PreviousCrime / education (and similar) into numbers; bad values become NA."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_text_scores(frame):
    """Add word counts and theme rates (hits per 100 words)."""
    out = frame.copy()
    tokens = out["LastStatement"].map(tokenize)
    out["word_count"] = tokens.map(len)
    out["declined"] = out["LastStatement"].map(is_declined)
    out.loc[out["declined"], "word_count"] = 0

    remorse = tokens.map(lambda words: theme_hits(words, REMORSE_EXACT, REMORSE_PREFIXES))
    gratitude = tokens.map(
        lambda words: theme_hits(words, GRATITUDE_LOVE_EXACT, GRATITUDE_LOVE_PREFIXES)
    )
    family = tokens.map(lambda words: theme_hits(words, FAMILY_EXACT, FAMILY_PREFIXES))
    religion = tokens.map(lambda words: theme_hits(words, RELIGION_EXACT, RELIGION_PREFIXES))

    out["remorse_hits"] = remorse
    out["gratitude_love_hits"] = gratitude
    out["family_hits"] = family
    out["religion_hits"] = religion
    out["apology_hits"] = remorse

    denom = out["word_count"].astype(float).where(out["word_count"] > 0)
    out["remorse_rate"] = (remorse.astype(float) / denom * 100).fillna(0.0)
    out["apology_rate"] = out["remorse_rate"]
    out["gratitude_love_rate"] = (gratitude.astype(float) / denom * 100).fillna(0.0)
    out["family_rate"] = (family.astype(float) / denom * 100).fillna(0.0)
    out["religion_rate"] = (religion.astype(float) / denom * 100).fillna(0.0)
    return out


def label_prior_groups(df):
    """Keep rows with a known PreviousCrime value and label No prior / Prior."""
    labeled = df.dropna(subset=["PreviousCrime"]).copy()
    labeled["group"] = labeled["PreviousCrime"].map({0.0: "No prior", 1.0: "Prior"})
    labeled = labeled[labeled["group"].notna()].copy()
    return labeled


def build_model_frame(labeled):
    """Frame used by the two original OLS models."""
    model_df = labeled[
        [
            "PreviousCrime",
            "apology_rate",
            "religion_rate",
            "remorse_rate",
            "gratitude_love_rate",
            "family_rate",
            "group",
            "declined",
            "Age",
            "EducationLevel",
            "Race",
        ]
    ].copy()
    model_df["prior_crime"] = model_df["PreviousCrime"].astype(int)
    for col in THEME_RATES:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").astype(float)
    return model_df


def fit_prior_models(model_df):
    """OLS: apology_rate ~ prior_crime and religion_rate ~ prior_crime."""
    x = sm.add_constant(model_df["prior_crime"].astype(float))
    apology_model = sm.OLS(model_df["apology_rate"].astype(float), x).fit()
    religion_model = sm.OLS(model_df["religion_rate"].astype(float), x).fit()
    return {"apology": apology_model, "religion": religion_model}


def prepare_demographic_frame(model_df):
    """Drop missing age/education and keep White / Black / Hispanic (White = reference)."""
    demo = model_df.dropna(subset=["Age", "EducationLevel", "Race"]).copy()
    demo = demo[demo["Race"].isin(["White", "Black", "Hispanic"])]
    demo["Race"] = pd.Categorical(demo["Race"], categories=["White", "Black", "Hispanic"])
    return demo


def demographic_design_matrix(demo):
    x_demo = pd.get_dummies(
        demo[["prior_crime", "Age", "EducationLevel", "Race"]],
        drop_first=True,
    ).astype(float)
    return sm.add_constant(x_demo)


def fit_theme_models(demo):
    """OLS: each theme rate ~ prior_crime + Age + EducationLevel + Race."""
    x_demo = demographic_design_matrix(demo)
    models = {}
    for outcome in ("remorse_rate", "gratitude_love_rate", "family_rate", "religion_rate"):
        models[outcome] = sm.OLS(demo[outcome].astype(float), x_demo).fit()
    return models


def plot_prior_crime(model_df, out_path):
    """Two-panel boxplot + strip: apology and religion by prior record."""
    sns.set_theme(style="ticks", context="notebook")
    plot_df = model_df.copy()
    plot_df["Apology / remorse"] = plot_df["apology_rate"]
    plot_df["Religious language"] = plot_df["religion_rate"]
    panels = [
        ("Apology / remorse", {"No prior": "#D7E2B4", "Prior": "#5E7A32"}, "#3F5320"),
        ("Religious language", {"No prior": "#F3D6E0", "Prior": "#B56B86"}, "#7A3F56"),
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
            hue_order=["No prior", "Prior"],
            order=["No prior", "Prior"],
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
            hue_order=["No prior", "Prior"],
            order=["No prior", "Prior"],
            palette=palette,
            legend=False,
            jitter=0.18,
            size=4.5,
            alpha=0.55,
            linewidth=0.6,
            edgecolor="white",
            ax=ax,
        )
        ax.set_title(col, pad=10, color=edge)
        ax.set_xlabel("")
        ax.set_ylabel("Hits per 100 words")
        sns.despine(ax=ax, trim=True)
        ax.spines["left"].set_color("#C9C3B8")
        ax.spines["bottom"].set_color("#C9C3B8")
        ax.yaxis.grid(True, color="#E6E1D8", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Last-statement language by prior criminal record",
        y=1.03,
        fontsize=16,
        fontweight="semibold",
        color="#2F2B26",
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_themes(demo, out_path):
    """Four-panel boxplot + strip: theme rates by race."""
    theme_plot = demo.copy()
    theme_plot["Remorse"] = theme_plot["remorse_rate"]
    theme_plot["Gratitude and love"] = theme_plot["gratitude_love_rate"]
    theme_plot["Family"] = theme_plot["family_rate"]
    theme_plot["Religion"] = theme_plot["religion_rate"]
    theme_panels = [
        ("Remorse", "#5E7A32"),
        ("Gratitude and love", "#B56B86"),
        ("Family", "#3D6B8A"),
        ("Religion", "#8A6A2F"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 5.0), facecolor="#FBF9F6")
    fig.patch.set_facecolor("#FBF9F6")
    race_order = ["White", "Black", "Hispanic"]
    race_palette = {"White": "#E8E0D2", "Black": "#8A7E6A", "Hispanic": "#C4B49A"}
    for ax, (col, edge) in zip(axes, theme_panels):
        ax.set_facecolor("#FBF9F6")
        sns.boxplot(
            data=theme_plot,
            x="Race",
            y=col,
            hue="Race",
            hue_order=race_order,
            order=race_order,
            palette=race_palette,
            width=0.6,
            linewidth=1.1,
            fliersize=0,
            legend=False,
            ax=ax,
        )
        for patch in ax.patches:
            patch.set_edgecolor(edge)
            patch.set_linewidth(1.1)
        sns.stripplot(
            data=theme_plot,
            x="Race",
            y=col,
            hue="Race",
            hue_order=race_order,
            order=race_order,
            palette=race_palette,
            legend=False,
            jitter=0.18,
            size=3.5,
            alpha=0.45,
            linewidth=0.5,
            edgecolor="white",
            ax=ax,
        )
        ax.set_title(col, pad=10, color=edge)
        ax.set_xlabel("")
        ax.set_ylabel("Hits per 100 words")
        sns.despine(ax=ax, trim=True)
        ax.spines["left"].set_color("#C9C3B8")
        ax.spines["bottom"].set_color("#C9C3B8")
        ax.yaxis.grid(True, color="#E6E1D8", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Last-word themes by race",
        y=1.03,
        fontsize=16,
        fontweight="semibold",
        color="#2F2B26",
    )
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def run_pipeline(data_path=None, plot_dir=None):
    """Full load → score → model → plot pipeline used by the system test."""
    data_path = Path(data_path) if data_path is not None else DATA_PATH
    plot_dir = Path(plot_dir) if plot_dir is not None else data_path.parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_statements(data_path)
    df = coerce_numeric(df)
    df = add_text_scores(df)
    labeled = label_prior_groups(df)
    model_df = build_model_frame(labeled)
    prior_models = fit_prior_models(model_df)
    demo = prepare_demographic_frame(model_df)
    theme_models = fit_theme_models(demo)
    prior_plot = plot_prior_crime(model_df, plot_dir / "apology_religion.png")
    theme_plot = plot_themes(demo, plot_dir / "last_statement_themes.png")
    return {
        "df": df,
        "labeled": labeled,
        "model_df": model_df,
        "demo": demo,
        "prior_models": prior_models,
        "theme_models": theme_models,
        "prior_plot": prior_plot,
        "theme_plot": theme_plot,
    }
