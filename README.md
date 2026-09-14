# ML-Assisted Query Plan Reuse

Dataset, generator and model comparison for a framework that reuses query
execution plans in a multi-tiered persistent storage system. Queries are
normalised into relational algebra, features are extracted from the normal form
and from each candidate execution plan, and the plan of the nearest stored
template is reused.

The dataset covers 15 TPC-H queries over four storage tiers (HDD, SSD,
high-speed SSD, NVMe): **180 query instances and 4836 candidate execution
plans**.

## Quick start

```bash
pip install -r requirements.txt
python dataset/generate_dataset.py    # regenerate the CSVs
python models/compare_models.py       # compare models, write results
```

The CSVs are committed, so the second command works on a fresh clone without
the first.

## Layout

```
dataset/
    generate_dataset.py      the generator: queries, cost model, plan space
    DATA_DICTIONARY.md       every column explained
    data_dictionary.csv      the same, machine readable
    query_features.csv       180 rows, one per query instance
    plan_features.csv        4836 rows, one per candidate plan
    template_dataset.csv     the two joined, ready for training
    instance_provenance.csv  how each instance was generated, not features
models/
    compare_models.py        model comparison across three tasks
    results/                 committed output, stamped with the commit
notebooks/                   exploratory work
```

## Why one repository

The dataset is generated rather than collected: changing a constant in
`generate_dataset.py` changes every number in it, and therefore every result.
Keeping the generator, the data and the results in one repository means they
move together in one commit, and a result can always be reproduced by checking
out the commit stamped in `models/results/model_comparison.csv`.

## The dataset

Features are split the way the paper splits them.

**Query features** come from the normal form `N(Q)`, so every instance of a
query has identical structural values — that is the property the framework
depends on. They divide into structural features, read off the normalised
expression, and statistical features, read from the catalogue.

**Plan features** come from the physical plan and are not normalised, because
the differences between candidates are exactly what the framework selects
between. They cover operation counts, pipelining and parallelism, per-tier data
volumes, and total data movement.

`DATA_DICTIONARY.md` explains every column. Each carries one of four roles —
`id`, `feature`, `target`, `metadata` — so feature selection can be done from
the dictionary rather than a hand-maintained list:

```python
dd = pd.read_csv('dataset/data_dictionary.csv')
features = dd[dd.role == 'feature'].column.unique()
```

## Tasks

| Task | Target | Type |
|---|---|---|
| `strategy` | `best_strategy_tag` | multiclass — which plan strategy wins |
| `optimal` | `optimal` | binary — is this candidate the cheapest |
| `penalty` | `cost_ratio_to_best` | regression — cost penalty of reuse |

```bash
python models/compare_models.py --task strategy
python models/compare_models.py --folds 10
```

## Evaluation

Every task uses `GroupKFold` on `query_group_id`. Instances of one group share
a normal form and therefore have identical structural features; a random split
would put near-identical rows on both sides and inflate the scores to the point
of meaninglessness.

Two things in the output need care.

**`optimal` is heavily imbalanced.** About 96% of candidates are not optimal, so
a model that always answers "no" scores 0.96 accuracy and 0.00 F1. Report F1 and
treat the majority baseline as the floor.

**`penalty` currently has negative R² for every model.** The target runs from
1.0 to about 37 with most mass near 1 and a long tail, so squared error is
dominated by a few extreme candidates. Predicting the log of the ratio is the
usual fix and is worth trying before concluding the task is not learnable.

## Provenance

The 180 instances come from 15 queries under three table-size settings and four
storage placements — `baseline`, the deployed configuration, plus three variants
standing for the same database after data migration. Those two axes are recorded
in `instance_provenance.csv` but deliberately kept out of the feature tables: a
query arriving at runtime carries no such label. Their effect reaches a model
only through measurable columns such as `input_mb`, `tier_min`, `tier_max` and
the per-tier volumes.

## What is measured and what is assumed

Tier read and write speeds, device counts, and the table sizes and home tiers
come from the measured storage configuration. The queries are the TPC-H
benchmark queries.

Tier capacities, the transient memory buffer, the random-access penalty for
index probes, and predicate selectivities are estimates. All are named constants
at the top of `dataset/generate_dataset.py`, and `DATA_DICTIONARY.md` lists them
with the effect each has.

## Citation

<!-- Add your name and the paper reference here before publishing, e.g.

    N. H. Aung, "<paper title>", ICECET 2027.

-->
