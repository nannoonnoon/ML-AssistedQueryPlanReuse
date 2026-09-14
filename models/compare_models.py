#!/usr/bin/env python3
"""
Compare models on the plan-reuse dataset.

Tasks
-----
    strategy   multiclass: which plan strategy wins for this query instance
    optimal    binary:     is this candidate the cheapest of its instance
    penalty    regression: cost of this candidate over the best cost

Every task is scored with GroupKFold on query_group_id. Instances of one group
share a normal form and therefore have identical structural features, so a
random split would place near-identical rows on both sides and the score would
be meaningless.

Results go to models/results/ as CSV, stamped with the git commit that
produced them, so any number in the paper can be traced back.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'dataset')
RESULTS = os.path.join(HERE, 'results')


def load():
    path = os.path.join(DATA, 'template_dataset.csv')
    if not os.path.exists(path):
        print('dataset not found. Generate it first:\n'
              '    python dataset/generate_dataset.py', file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    dd = pd.read_csv(os.path.join(DATA, 'data_dictionary.csv'))
    return df, dd, dataset_stamp()


def dataset_stamp():
    import subprocess
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'not-a-git-checkout'


def feature_columns(df, dd):
    named = set(dd.loc[dd.role == 'feature', 'column'])
    cols = []
    for c in df.columns:
        stem = c[:-5] if c.endswith('_plan') else \
               c[:-6] if c.endswith('_query') else c
        if stem in named and df[c].dtype.kind in 'if':
            cols.append(c)
    return cols


CLASSIFIERS = {
    'baseline (most frequent)': DummyClassifier(strategy='most_frequent'),
    'kNN k=3': make_pipeline(StandardScaler(), KNeighborsClassifier(3)),
    'kNN k=5': make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    'kNN k=10': make_pipeline(StandardScaler(), KNeighborsClassifier(10)),
    'decision tree': DecisionTreeClassifier(random_state=0),
    'logistic regression': make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000)),
    'random forest': RandomForestClassifier(n_estimators=200, random_state=0,
                                            n_jobs=-1),
    'gradient boosting': GradientBoostingClassifier(random_state=0),
}

REGRESSORS = {
    'baseline (mean)': DummyRegressor(strategy='mean'),
    'kNN k=5': make_pipeline(StandardScaler(), KNeighborsRegressor(5)),
    'ridge': make_pipeline(StandardScaler(), Ridge()),
    'random forest': RandomForestRegressor(n_estimators=200, random_state=0,
                                           n_jobs=-1),
}

try:
    from xgboost import XGBClassifier, XGBRegressor
    CLASSIFIERS['xgboost'] = XGBClassifier(
        n_estimators=200, random_state=0, verbosity=0)
    REGRESSORS['xgboost'] = XGBRegressor(
        n_estimators=200, random_state=0, verbosity=0)
except ImportError:
    print('xgboost not installed, skipping it (pip install xgboost)\n',
          file=sys.stderr)

TASKS = {
    'strategy': dict(target='best_strategy_tag', kind='clf',
                     scoring=['accuracy', 'f1_macro'],
                     desc='which plan strategy wins'),
    'optimal': dict(target='optimal', kind='clf',
                    scoring=['accuracy', 'f1'],
                    desc='is this candidate the optimal one'),
    'penalty': dict(target='cost_ratio_to_best', kind='reg',
                    scoring=['neg_mean_absolute_error', 'r2'],
                    desc='cost penalty relative to the best plan'),
}


def run_task(name, df, feats, folds):
    spec = TASKS[name]
    y = df[spec['target']].values
    X = df[feats].values
    groups = df.query_group_id.values
    models = CLASSIFIERS if spec['kind'] == 'clf' else REGRESSORS
    cv = GroupKFold(n_splits=folds)

    print(f"\n=== {name}: {spec['desc']} ===")
    print(f"target {spec['target']}   rows {len(df)}   features {len(feats)}   "
          f"groups {df.query_group_id.nunique()}")
    rows = []
    for label, model in models.items():
        cvres = cross_validate(model, X, y, groups=groups, cv=cv,
                               scoring=spec['scoring'], n_jobs=-1)
        row = dict(task=name, model=label)
        parts = []
        for s in spec['scoring']:
            v = cvres[f'test_{s}']
            row[f'{s}_mean'] = round(float(np.mean(v)), 4)
            row[f'{s}_std'] = round(float(np.std(v)), 4)
            shown = -np.mean(v) if s.startswith('neg_') else np.mean(v)
            parts.append(f"{s.replace('neg_', ''):24s} {shown:7.4f} "
                         f"+/- {np.std(v):.4f}")
        rows.append(row)
        print(f'  {label:26s} ' + '   '.join(parts))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', choices=list(TASKS) + ['all'], default='all')
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    df, dd, version = load()
    feats = feature_columns(df, dd)
    os.makedirs(RESULTS, exist_ok=True)
    print(f'commit: {version}')
    print(f'features: {len(feats)}')

    names = list(TASKS) if args.task == 'all' else [args.task]
    out = pd.concat([run_task(n, df, feats, args.folds) for n in names],
                    ignore_index=True)
    out['commit'] = version
    out['n_folds'] = args.folds
    path = os.path.join(RESULTS, 'model_comparison.csv')
    out.to_csv(path, index=False)
    print(f'\nwrote {path}')

    with open(os.path.join(RESULTS, 'feature_list.txt'), 'w') as fh:
        fh.write('\n'.join(feats) + '\n')


if __name__ == '__main__':
    main()
