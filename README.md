# Week 2 Mini-Assignment: ASD gut microbiome

Alissa Rivero

This repository has the three required pieces:

1. A **Python data analysis script** — [`question1.py`](question1.py) (same analysis in [`question1.ipynb`](question1.ipynb))
2. This **README**
3. A **Rust Jupyter notebook** — [`question2.ipynb`](question2.ipynb)

The last section of the Python script (and notebook) also times **Pandas vs Polars** on the same pipeline.

**Research questions**

1. Is ASD associated with more or less **good** gut bacteria?
2. Is ASD associated with more or less **bad** gut bacteria?

## Dataset

[Human Gut Microbiome with ASD (Kaggle)](https://www.kaggle.com/datasets/antaresnyc/human-gut-microbiome-with-asd/data)

The file used here is `ASD meta abundance.csv`: shotgun metagenome abundances from Dan et al., 2020 (*Gut Microbes*).

| | |
|---|---|
| Rows | 5,619 species (`g__Genus;s__Species`) |
| Columns | `Taxonomy` plus 60 stool samples |
| Values | Integer read counts (not percentages) |
| Groups | `A*` = 30 ASD samples; `B*` = 30 typically developing controls |
| Missing values | None (zeros mean the taxon was not detected) |

About two-thirds of the cells are zero, and sample totals differ, so group comparisons use **relative abundance** (each sample’s counts divided by that sample’s total).

A second file, `GSE113690_Autism_16S_rRNA_OTU_assignment_and_abundance.csv`, is the 16S rRNA OTU table from the same study. It has more samples and is not used in the analysis.

Papers: [GSE113690](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113690) (16S) and [GSE113540](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113540) (metagenomes). Beneficial vs harmful taxa used in the models follow the paper’s categorizations.

## How to run the Python analysis

```bash
pip install pandas statsmodels matplotlib seaborn polars
python question1.py
```

Or open `question1.ipynb` and run all cells. The CSV files must stay in this same folder. If you see `ModuleNotFoundError`, the notebook kernel is a different Python than the one where you installed packages.

## What the Python analysis does

1. Load and inspect `ASD meta abundance.csv`.
2. Split columns into ASD (`A*`) and control (`B*`).
3. Reshape to long format and use `groupby()` for mean / count / std by group and by genus.
4. Compare genera side by side with percent difference, keep genera above 0.01%, and split into higher vs lower in ASD.
5. Build two per-sample scores (sum of relative abundance):
   - **good bacteria** — *Bifidobacterium*, *Lactobacillus*, *Faecalibacterium*, *Roseburia*, *Ruminococcus* (except *R. gnavus*), *Bacteroides*, *Prevotella*, *Oxalobacter formigenes*
   - **bad bacteria** — *C. difficile*, *E. coli*, *Salmonella*, *Shigella*, *Klebsiella*, *Serratia*, *Pseudomonas*, *Campylobacter*, *R. gnavus*, *Desulfovibrio*, plus mapped Enterobacteriaceae / Fusobacteriaceae / Veillonellaceae genera
6. Fit two OLS models with autism as the predictor (`1` = ASD, `0` = control):
   - `good_bacteria ~ autism`
   - `bad_bacteria ~ autism`
7. Plot both scores as boxplots (with sample points) for Control vs ASD.
8. Time the same relative-abundance + genus mean pipeline in **Pandas** and **Polars**.

*Helicobacter hepaticus* is listed in the paper but is not in this table.

## Question 2: Rust notebook

[`question2.ipynb`](question2.ipynb) is the class `rust_vs_python_intro.ipynb` notebook, run with the **Rust** kernel, plus extra cells at the bottom that experiment with ownership using taxon names from this dataset.

1. Install the Rust Jupyter kernel (`evcxr_jupyter --install`) if it is not already there.
2. Open `question2.ipynb` and pick **Rust** (not Python). If `let` is a `SyntaxError`, you are still on Python.
3. Run all cells from top to bottom. A few cells are supposed to fail — that is the ownership point.

What I changed:

- Filled in the class “Your turn” cells (name, movie, cutoff, loop, `mut`, `.clone()`).
- Added extra cells that **move** a `Vec` of taxa, **clone** it, pass it by value vs `&` borrow, and try a read + write at the same time (that cell is supposed to fail).

## Files

| File | Role |
|---|---|
| `question1.py` | Data analysis Python script (required) |
| `question1.ipynb` | Same analysis as a notebook, plus the Polars timing |
| `question2.ipynb` | Modified Rust Jupyter notebook (required) |
| `README.md` | This file (required) |
| `ASD meta abundance.csv` | Metagenome abundance table used in the analysis |
| `GSE113690_Autism_16S_rRNA_OTU_assignment_and_abundance.csv` | 16S OTU table (not used) |
| `good_bad_bacteria.png` | Figure written by `question1.py` |
