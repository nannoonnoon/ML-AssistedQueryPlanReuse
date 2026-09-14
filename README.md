# Model Selection for Query Plan Reuse

Model comparison for a framework that reuses query execution plans in a
multi-tiered storage system. The dataset lives in a separate repository and is
pulled in at a pinned version.

## Setup

```bash
pip install -r requirements.txt
python src/fetch_data.py          # download the pinned dataset into data/
python src/compare_models.py      # run every task, write results/
```

Before the first fetch, open `src/fetch_data.py` and set:

```python
DATASET_REPO = 'YOUR-USERNAME/plan-reuse-dataset'
DATASET_TAG  = 'v1.0'
```

While developing both repositories side by side, skip the download:

```bash
python src/fetch_data.py --local ../plan-reuse-dataset
```

## Why the dataset is pinned

The dataset is generated, not collected: changing a constant in its generator
changes every number in it. If this repository simply held a copy of the CSVs,
there would be no way to tell months later which version produced the results
in the paper.

So `data/` is gitignored, `fetch_data.py` pulls a specific tag, and the tag is
written to `data/VERSION.txt` and copied into every results file. Any number
reported here can be traced to the exact data behind it.

**When you bump `DATASET_TAG`, re-run every experiment.** Results from one tag
are not comparable with results from another.

## Tasks

| Task | Target | Type |
|---|---|---|
| `strategy` | `best_strategy_tag` | multiclass — which plan strategy wins |
| `optimal` | `optimal` | binary — is this candidate the cheapest |
| `penalty` | `cost_ratio_to_best` | regression — cost penalty of reuse |

```bash
python src/compare_models.py --task strategy
python src/compare_models.py --folds 10
```

## Evaluation

Every task uses `GroupKFold` on `query_group_id`. Instances of one group share
a normal form and therefore have identical structural features; a random split
would put near-identical rows on both sides and the scores would be inflated to
the point of meaninglessness.

Features come from the `role` column of `data_dictionary.csv` rather than a
hard-coded list, so a column added to the dataset is picked up automatically and
an id, target or metadata column can never leak in by accident.

## Reading the results

Two things in the output need care.

**`optimal` is heavily imbalanced.** About 96% of candidates are not optimal, so
a model that always answers "no" scores 0.96 accuracy and 0.00 F1. Accuracy is
useless here — report F1, and treat the F1 of the majority baseline (zero) as
the floor to beat.

**`penalty` currently has negative R² for every model.** The target ranges from
1.0 to roughly 37, with most mass near 1 and a long tail, so squared error is
dominated by a few extreme candidates. Predicting `log(cost_ratio_to_best)` and
reporting error in log space is the usual fix, and is worth trying before
concluding the task is not learnable.

## Layout

```
src/fetch_data.py       pull the dataset at a pinned version
src/compare_models.py   run the tasks, write results/
data/                   fetched dataset (gitignored)
results/                model_comparison.csv, feature_list.txt (committed)
notebooks/              exploratory work
```

`results/` is committed so the numbers in the paper are traceable without
re-running anything.

## Related

Dataset and generator: `https://github.com/YOUR-USERNAME/plan-reuse-dataset`
