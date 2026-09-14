#!/usr/bin/env python3
"""
Generate the template dataset for the query-plan-reuse framework.

Run
---
    python generate_dataset.py                # write beside this script
    python generate_dataset.py /path/to/dir   # write there
    python generate_dataset.py --no-excel     # CSV only (no openpyxl needed)

Needs Python 3 with pandas; openpyxl only for the .xlsx. No input files:
every query, table size and device speed is a constant below. The generator
is deterministic, so the same code always produces the same dataset.

Outputs
-------
    query_features.csv      one row per query instance     
    plan_features.csv       one row per candidate plan     
    template_dataset.csv    the two joined, ready for ML
    template_dataset.xlsx   the same three as worksheets
    instance_provenance.csv how each instance was generated (NOT features)
    data_dictionary.csv     every column, its meaning, type and role

Sources
-------
    tier speeds, device counts      measured storage configuration
    table sizes and placement       TPC-H database as deployed
    the 15 queries                  TPC-H benchmark
"""

import argparse
import math
import os
import sys

import pandas as pd

# =====================================================================
# 1. STORAGE CONFIGURATION
# =====================================================================
# Read and write speeds in MB/s of the measured storage configuration.
READ  = {0: 240.0, 1: 540.0, 2: 1000.0, 3: 5000.0}
WRITE = {0: 230.0, 1: 530.0, 2: 950.0,  3: 4500.0}

# Devices per tier. l3 is a single NVMe device, so an operation placed on
# l3 cannot be partitioned across devices.
NDEV = {0: 3, 1: 3, 2: 3, 3: 1}

# --- assumptions, not measurements -----------------------------------
# Usable capacity per tier in MB. Not measured; estimated. These decide
# when a plan can place an intermediate result on the fast tiers.
CAP = {0: 1e9, 1: 1e9, 2: 6000.0, 3: 2000.0}
# Transient memory available for pipelining and for a hash build side.
BUF = 1024.0
# Index probes are seek-bound, so a given volume read at random costs this
# multiple of the same volume read sequentially.
RAND = 3.0

# =====================================================================
# 2. BASE TABLES  (size in MB, home tier of the deployed database)
# =====================================================================
TABLES = {
    'LINEITEM': (6409, 1),   # 2563 + 2563 + 1283 across l1.d1..d3
    'ORDERS':   (1487, 1),   # 992 + 495 across l1.d2, l1.d3
    'PARTSUPP': (1099, 0),
    'PART':     (296,  0),
    'CUSTOMER': (256,  0),
    'SUPPLIER': (15,   0),
    'NATION':   (1,    0),
    'REGION':   (1,    0),
}

# =====================================================================
# 3. THE 15 QUERIES  (TPC-H)
# =====================================================================
# sel    local selection predicates as (relation, selectivity)
# joins  left-deep join order after normalisation, relations in lexicographic
#        order per rule j1, as (left relations, right relation, selectivity)
# subq   relations read by a subquery block, as (relation, selectivity)
QUERIES = {}


def define(qid, rels, sel, joins, n_agg, n_group, n_sort,
           n_anti=0, depth=1, n_terms=1, subq=None, cls=''):
    QUERIES[qid] = dict(qid=qid, rels=rels, sel=sel, joins=joins,
                        n_agg=n_agg, n_group=n_group, n_sort=n_sort,
                        n_anti=n_anti, depth=depth, n_terms=n_terms,
                        subq=subq or [], cls=cls)


define('Q1', ['LINEITEM'], [('LINEITEM', 0.98)], [], 8, 2, 2,
       cls='aggregate and group over one relation')

define('Q2', ['NATION', 'PART', 'PARTSUPP', 'REGION', 'SUPPLIER'],
       [('PART', 0.0040), ('REGION', 0.20)],
       [(['PART'], 'PARTSUPP', 5e-7),
        (['PART', 'PARTSUPP'], 'SUPPLIER', 1e-5),
        (['PART', 'PARTSUPP', 'SUPPLIER'], 'NATION', 0.04),
        (['NATION', 'PART', 'PARTSUPP', 'SUPPLIER'], 'REGION', 0.20)],
       1, 0, 4, depth=2,
       subq=[('PARTSUPP', 5e-7), ('SUPPLIER', 1e-5),
             ('NATION', 0.04), ('REGION', 0.20)],
       cls='five-way join with correlated aggregate subquery')

define('Q3', ['CUSTOMER', 'LINEITEM', 'ORDERS'],
       [('CUSTOMER', 0.20), ('ORDERS', 0.48), ('LINEITEM', 0.54)],
       [(['CUSTOMER'], 'ORDERS', 1e-7),
        (['CUSTOMER', 'ORDERS'], 'LINEITEM', 1e-7)],
       1, 3, 2, cls='three-way join, group, sort, limit')

define('Q4', ['LINEITEM', 'ORDERS'],
       [('ORDERS', 0.038), ('LINEITEM', 0.63)], [], 1, 1, 1, depth=2,
       subq=[('LINEITEM', 0.63)],
       cls='semi-join from EXISTS, group, sort')

define('Q5', ['CUSTOMER', 'LINEITEM', 'NATION', 'ORDERS', 'REGION', 'SUPPLIER'],
       [('REGION', 0.20), ('ORDERS', 0.15)],
       [(['CUSTOMER'], 'ORDERS', 1e-7),
        (['CUSTOMER', 'ORDERS'], 'LINEITEM', 1e-7),
        (['CUSTOMER', 'LINEITEM', 'ORDERS'], 'SUPPLIER', 1e-5),
        (['CUSTOMER', 'LINEITEM', 'ORDERS', 'SUPPLIER'], 'NATION', 0.04),
        (['CUSTOMER', 'LINEITEM', 'NATION', 'ORDERS', 'SUPPLIER'], 'REGION', 0.20)],
       1, 1, 1, cls='six-way join, group, sort')

define('Q6', ['LINEITEM'], [('LINEITEM', 0.019)], [], 1, 0, 0,
       cls='selective filter with one aggregate')

define('Q7', ['CUSTOMER', 'LINEITEM', 'NATION', 'ORDERS', 'SUPPLIER'],
       [('LINEITEM', 0.30), ('NATION', 0.08)],
       [(['LINEITEM'], 'ORDERS', 1e-7),
        (['LINEITEM', 'ORDERS'], 'CUSTOMER', 1e-7),
        (['CUSTOMER', 'LINEITEM', 'ORDERS'], 'SUPPLIER', 1e-5),
        (['CUSTOMER', 'LINEITEM', 'ORDERS', 'SUPPLIER'], 'NATION', 0.04)],
       1, 3, 3, depth=2, cls='derived table, five-way join, group, sort')

define('Q8', ['CUSTOMER', 'LINEITEM', 'NATION', 'ORDERS', 'PART', 'REGION', 'SUPPLIER'],
       [('PART', 0.0002), ('REGION', 0.20), ('ORDERS', 0.30)],
       [(['LINEITEM'], 'PART', 5e-7),
        (['LINEITEM', 'PART'], 'SUPPLIER', 1e-5),
        (['LINEITEM', 'PART', 'SUPPLIER'], 'ORDERS', 1e-7),
        (['LINEITEM', 'ORDERS', 'PART', 'SUPPLIER'], 'CUSTOMER', 1e-7),
        (['CUSTOMER', 'LINEITEM', 'ORDERS', 'PART', 'SUPPLIER'], 'NATION', 0.04),
        (['CUSTOMER', 'LINEITEM', 'NATION', 'ORDERS', 'PART', 'SUPPLIER'], 'REGION', 0.20)],
       2, 1, 1, depth=2, cls='derived table, seven-way join, ratio aggregate')

define('Q9', ['LINEITEM', 'NATION', 'ORDERS', 'PART', 'PARTSUPP', 'SUPPLIER'],
       [('PART', 0.05)],
       [(['LINEITEM'], 'PART', 5e-7),
        (['LINEITEM', 'PART'], 'SUPPLIER', 1e-5),
        (['LINEITEM', 'PART', 'SUPPLIER'], 'PARTSUPP', 5e-7),
        (['LINEITEM', 'PART', 'PARTSUPP', 'SUPPLIER'], 'ORDERS', 1e-7),
        (['LINEITEM', 'ORDERS', 'PART', 'PARTSUPP', 'SUPPLIER'], 'NATION', 0.04)],
       1, 2, 2, depth=2, cls='derived table, six-way join, group, sort')

define('Q10', ['CUSTOMER', 'LINEITEM', 'NATION', 'ORDERS'],
       [('ORDERS', 0.038), ('LINEITEM', 0.25)],
       [(['CUSTOMER'], 'ORDERS', 1e-7),
        (['CUSTOMER', 'ORDERS'], 'LINEITEM', 1e-7),
        (['CUSTOMER', 'LINEITEM', 'ORDERS'], 'NATION', 0.04)],
       1, 7, 1, cls='four-way join, wide grouping, sort, limit')

define('Q11', ['NATION', 'PARTSUPP', 'SUPPLIER'], [('NATION', 0.04)],
       [(['PARTSUPP'], 'SUPPLIER', 1e-5),
        (['PARTSUPP', 'SUPPLIER'], 'NATION', 0.04)],
       1, 1, 1, depth=2,
       subq=[('PARTSUPP', 1.0), ('SUPPLIER', 1e-5), ('NATION', 0.04)],
       cls='three-way join with HAVING against a scalar subquery')

define('Q12', ['LINEITEM', 'ORDERS'], [('LINEITEM', 0.011)],
       [(['LINEITEM'], 'ORDERS', 1e-7)], 2, 1, 1,
       cls='two-way join with conditional aggregates')

define('Q13', ['CUSTOMER', 'ORDERS'], [('ORDERS', 0.97)],
       [(['CUSTOMER'], 'ORDERS', 1e-7)], 2, 2, 2, depth=2,
       cls='outer join in a derived table, two levels of grouping')

define('Q14', ['LINEITEM', 'PART'], [('LINEITEM', 0.013)],
       [(['LINEITEM'], 'PART', 5e-7)], 2, 0, 0,
       cls='two-way join with a ratio aggregate')

define('Q15', ['LINEITEM', 'PART'], [('PART', 0.0002)],
       [(['LINEITEM'], 'PART', 5e-7)], 1, 0, 0, depth=2,
       subq=[('LINEITEM', 1.0)],
       cls='two-way join with correlated aggregate subquery')

# =====================================================================
# 4. GENERATION AXES
# =====================================================================
# These multiply the number of query instances. They are NOT features:
# a new query arriving at runtime does not carry a "scale factor" or a
# "placement" label. Their effect reaches the model only through the
# measurable columns (input_mb, tier_min, tier_max, n_large, n_small and
# the per-tier volumes), which is what a real system would observe.
# They are written to instance_provenance.csv for analysis only.
SCALES = {'SF1': 1.0, 'SF3': 3.0, 'SF10': 10.0}

PLACEMENTS = {
    # the configuration as deployed and measured
    'baseline': dict(TABLES),
    # hypothetical variants, standing for the same database after migration
    'cold':     {k: (v[0], 0) for k, v in TABLES.items()},
    'hot':      {k: (v[0], min(v[1] + 1, 2)) for k, v in TABLES.items()},
    'split':    {k: (v[0], 0 if v[0] > 1000 else 2) for k, v in TABLES.items()},
}

# =====================================================================
# 5. CANDIDATE PLAN SPACE
# =====================================================================
JOIN_ALG = {'nested_loop': 0, 'hash': 1, 'sort_merge': 2, 'index_nested_loop': 3}
ACCESS = {'seq': 0, 'index': 1}

def build_variants():
    """Enumerate the candidate plans an optimiser would consider.

    Four choices: the tier intermediates are written to, the partition
    count, the join algorithm, and whether operations are pipelined.
    Incoherent combinations are excluded: a sort-merge join carries its
    extra sorts, an index nested-loop join implies index access, and
    pipelining needs a target tier faster than the home tier.
    """
    out, tag = [], 0
    for itier, tname in [(None, 'home'), (2, 'l2'), (3, 'l3')]:
        for par in [0, 3]:            # no tier has more than three devices
            for alg, acc, extra_sort in [('nested_loop', 'seq', 0),
                                         ('hash', 'seq', 0),
                                         ('index_nested_loop', 'index', 0),
                                         ('sort_merge', 'seq', 2)]:
                for pipe in ([False, True] if itier is not None else [False]):
                    tag += 1
                    label = f"{tname}, {'p3' if par else 'serial'}, {alg}"
                    if pipe:
                        label += ', pipelined'
                    out.append(dict(tag=tag, name=label, itier=itier,
                                    pipe=pipe, par=par, jalg=alg,
                                    access=acc, extra_sort=extra_sort))
    return out


VARIANTS = build_variants()

# =====================================================================
# 6. SIZE PROPAGATION AND COST MODEL
# =====================================================================
STATE = dict(tables=TABLES, scale=1.0)


def base_size(rel):
    return float(STATE['tables'][rel][0]) * STATE['scale']


def home_tier(rel):
    return STATE['tables'][rel][1]


def after_filter(rel, spec):
    size = base_size(rel)
    for r, f in spec['sel']:
        if r == rel:
            size *= f
    return max(size, 0.5)


def logical_ops(spec):
    """Operations of the normalised plan, in execution order."""
    ops = []
    for rel in sorted(spec['rels']):
        kind = 'sel' if any(r == rel for r, _ in spec['sel']) else 'scan'
        ops.append(dict(kind=kind, rel=rel, rin=base_size(rel),
                        rout=after_filter(rel, spec), tier=home_tier(rel)))
    for rel, f in spec['subq']:
        ops.append(dict(kind='subq', rel=rel, rin=base_size(rel),
                        rout=max(base_size(rel) * f, 0.5), tier=home_tier(rel)))
    running = None
    for left, right, jsel in spec['joins']:
        lsize = running if running is not None else \
            sum(after_filter(r, spec) for r in left)
        rsize = after_filter(right, spec)
        out = min(max(lsize * rsize * jsel, 1.0), lsize + rsize)
        ops.append(dict(kind='join', rel=right, rin=(lsize, rsize),
                        rout=out, tier=None))
        running = out
    cur = running if running is not None else \
        sum(after_filter(r, spec) for r in spec['rels'])
    if spec['n_agg']:
        cur = max(cur * 0.02, 0.5) if spec['n_group'] else 0.5
        ops.append(dict(kind='agg', rel=None, rin=ops[-1]['rout'] if ops else cur,
                        rout=cur, tier=None))
    if spec['n_sort']:
        ops.append(dict(kind='sort', rel=None, rin=cur, rout=cur, tier=None))
    return ops


def join_read_volume(alg, left, right, out):
    """Read volume of a join, which is where the algorithms differ."""
    small, big = min(left, right), max(left, right)
    if alg == 'index_nested_loop':
        # only the matching inner rows, but read at random
        return big + RAND * max(out * 1.2, 0.5)
    if alg == 'hash':
        # the build side must fit in memory, otherwise the join spills
        return big + small + (2 * small if small > BUF else 0.0)
    if alg == 'sort_merge':
        return big + small
    # nested loop re-reads the inner relation once per block of the outer
    return small + max(1.0, math.ceil(small / BUF)) * big


def build_candidate(spec, v):
    ops = logical_ops(spec)
    vol = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    cost = read_total = write_total = 0.0
    n_cross = n_push = n_pipe = n_par = 0
    pipe_run = pipe_max = par_max = 0
    inter, tiers = [], set()
    counts = dict(scan=0, sel=0, subq=0, join=0, agg=0, sort=0)
    prev_dst = 1
    joined = False

    for i, op in enumerate(ops):
        counts[op['kind']] += 1
        last = (i == len(ops) - 1)

        # ---- source tier and read volume
        if op['kind'] in ('scan', 'sel', 'subq'):
            src = op['tier']
            rvol = op['rin']
            if v['access'] == 'index' and op['rout'] / op['rin'] < 0.05:
                rvol = RAND * max(op['rout'] * 1.2, 0.5)
        else:
            src = v['itier'] if v['itier'] is not None else prev_dst
            if op['kind'] == 'join':
                rvol = join_read_volume(v['jalg'], op['rin'][0],
                                        op['rin'][1], op['rout'])
                joined = True
            else:
                rvol = op['rin']

        # ---- destination tier, falling back when the result does not fit
        if v['itier'] is None:
            dst = op['tier'] if op['tier'] is not None else prev_dst
        else:
            dst = v['itier']
            while dst > 0 and op['rout'] > CAP[dst]:
                dst -= 1
        wvol = op['rout']

        # ---- partitioning, limited by the devices at the tiers touched
        p = min(v['par'], NDEV[src], NDEV[dst]) if v['par'] > 1 else 1
        parallel = p > 1 and op['kind'] in ('scan', 'sel', 'subq', 'join')
        if not parallel:
            p = 1
        else:
            n_par += 1
            par_max = max(par_max, p)

        # ---- pipelining, only if the result fits in transient memory
        pipelined = (v['pipe'] and joined and not last
                     and op['kind'] in ('join', 'agg') and wvol <= BUF)
        if pipelined:
            n_pipe += 1
            pipe_run += 1
            pipe_max = max(pipe_max, pipe_run)
        else:
            pipe_run = 0

        # ---- charge
        cost += (rvol / p) / READ[src]
        vol[src] += rvol
        read_total += rvol
        tiers.add(src)
        if not pipelined:
            cost += (wvol / p) / WRITE[dst]
            vol[dst] += wvol
            write_total += wvol
            tiers.add(dst)
        if dst != src:
            n_cross += 1
            if dst > src:
                n_push += 1
        if not last:
            inter.append(wvol)
        prev_dst = dst

    # sorts required by a sort-merge join
    for _ in range(v['extra_sort']):
        s = inter[-1] if inter else 1.0
        t = v['itier'] if v['itier'] is not None else 1
        cost += s / READ[t] + s / WRITE[t]
        vol[t] += 2 * s
        read_total += s
        write_total += s
        counts['sort'] += 1

    total_in = sum(base_size(r) for r in spec['rels'])
    return dict(
        n_ops=sum(counts.values()),
        n_joins=counts['join'],
        n_sel=counts['sel'] + counts['subq'],
        n_agg=counts['agg'],
        n_sort=counts['sort'],
        join_typ=JOIN_ALG[v['jalg']],
        access=ACCESS[v['access']],
        n_pipe=n_pipe, pipe_max=pipe_max, n_par=n_par, par_max=par_max,
        d_0=round(vol[0], 1), d_1=round(vol[1], 1),
        d_2=round(vol[2], 1), d_3=round(vol[3], 1),
        n_cross=n_cross, n_push=n_push,
        tier_span=(max(tiers) - min(tiers)) if tiers else 0,
        read_total=round(read_total, 1),
        write_total=round(write_total, 1),
        inter_max=round(max(inter) if inter else 0.0, 1),
        reduce=round(ops[-1]['rout'] / total_in, 6),
        cost=round(cost, 4),
    )


# =====================================================================
# 7. ASSEMBLY
# =====================================================================
def size_class(mb):
    return 'large' if mb > 2000 else 'medium' if mb > 200 else 'small'


def generate():
    qrows, prows, prov = [], [], []
    counter = {qid: 0 for qid in QUERIES}

    for sf, mult in SCALES.items():
        for pname, tables in PLACEMENTS.items():
            STATE['tables'], STATE['scale'] = tables, mult
            for qid, spec in QUERIES.items():
                counter[qid] += 1
                inst = f"{qid}_I{counter[qid]:02d}"
                prov.append(dict(instance_id=inst, query_group_id=qid,
                                 scale_factor=sf, placement=pname))

                sizes = {r: base_size(r) for r in spec['rels']}
                qrows.append(dict(
                    instance_id=inst, query_group_id=qid,
                    n_terms=spec['n_terms'], n_rel=len(spec['rels']),
                    n_joins=len(spec['joins']), n_anti=spec['n_anti'],
                    n_sel=len(spec['sel']), n_agg=spec['n_agg'],
                    n_group=spec['n_group'], n_sort=spec['n_sort'],
                    depth=spec['depth'],
                    size_class_max=size_class(max(sizes.values())),
                    n_large=sum(1 for v in sizes.values() if v > 2000),
                    n_small=sum(1 for v in sizes.values() if v <= 200),
                    tier_min=min(tables[r][1] for r in spec['rels']),
                    tier_max=max(tables[r][1] for r in spec['rels']),
                    input_mb=round(sum(sizes.values()), 1),
                    query_class=spec['cls'],
                ))

                cands = []
                for v in VARIANTS:
                    f = build_candidate(spec, v)
                    f.update(template_id=f"{inst}_T{v['tag']:02d}",
                             instance_id=inst, query_group_id=qid,
                             plan_strategy=v['name'], strategy_tag=v['tag'])
                    cands.append(f)

                # a query with no join gives identical plans for every join
                # algorithm; keep one representative per distinct vector
                seen, uniq = set(), []
                for c in cands:
                    key = tuple(c[k] for k in
                                ('n_ops', 'n_pipe', 'pipe_max', 'n_par',
                                 'par_max', 'd_0', 'd_1', 'd_2', 'd_3',
                                 'n_cross', 'n_push', 'tier_span',
                                 'read_total', 'write_total', 'cost'))
                    if key not in seen:
                        seen.add(key)
                        uniq.append(c)

                best = min(uniq, key=lambda c: c['cost'])
                for c in uniq:
                    c['optimal'] = int(c['template_id'] == best['template_id'])
                    c['plan_template_id'] = best['template_id']
                    c['best_strategy_tag'] = best['strategy_tag']
                    c['cost_ratio_to_best'] = round(c['cost'] / best['cost'], 4)
                prows.extend(uniq)

    qf = pd.DataFrame(qrows)
    pcols = ['template_id', 'instance_id', 'query_group_id', 'plan_strategy',
             'strategy_tag', 'n_ops', 'n_joins', 'n_sel', 'n_agg', 'n_sort',
             'join_typ', 'access', 'n_pipe', 'pipe_max', 'n_par', 'par_max',
             'd_0', 'd_1', 'd_2', 'd_3', 'n_cross', 'n_push', 'tier_span',
             'read_total', 'write_total', 'inter_max', 'reduce', 'cost',
             'cost_ratio_to_best', 'optimal', 'plan_template_id',
             'best_strategy_tag']
    pf = pd.DataFrame(prows)[pcols]
    return qf, pf, pd.DataFrame(prov)


# =====================================================================
# 8. DATA DICTIONARY
# =====================================================================
# role: id | feature | target | metadata
DICT_ROWS = [
    ('query_features', 'instance_id', 'string', 'id',
     'Identifier of one query instance.'),
    ('query_features', 'query_group_id', 'string', 'id',
     'The query this instance belongs to. All instances of a group share a '
     'normal form. Use this to split train and test.'),
    ('query_features', 'n_terms', 'int', 'feature',
     'Number of terms in the union of the normal form.'),
    ('query_features', 'n_rel', 'int', 'feature',
     'Number of distinct base relations referenced by the normal form.'),
    ('query_features', 'n_joins', 'int', 'feature',
     'Number of inner joins in the normal form.'),
    ('query_features', 'n_anti', 'int', 'feature',
     'Number of anti-joins and semi-joins, arising from EXCEPT, NOT EXISTS '
     'and NOT IN after rules t2 and the EXISTS translation.'),
    ('query_features', 'n_sel', 'int', 'feature',
     'Number of selection predicates over base attributes.'),
    ('query_features', 'n_agg', 'int', 'feature',
     'Number of aggregate functions computed.'),
    ('query_features', 'n_group', 'int', 'feature',
     'Number of grouping attributes.'),
    ('query_features', 'n_sort', 'int', 'feature',
     'Number of ordering attributes.'),
    ('query_features', 'depth', 'int', 'feature',
     'Maximum nesting depth of normalised subexpressions. 1 for a flat '
     'query, 2 when a join operand or an anti-join operand is itself a '
     'normal form.'),
    ('query_features', 'size_class_max', 'category', 'feature',
     'Size class of the largest input relation: small (<=200 MB), medium '
     '(<=2000 MB) or large. Encode before use.'),
    ('query_features', 'n_large', 'int', 'feature',
     'Number of input relations over 2000 MB.'),
    ('query_features', 'n_small', 'int', 'feature',
     'Number of input relations at or under 200 MB. Small relations can be '
     'held in transient memory, which is what makes a nested-loop join '
     'viable.'),
    ('query_features', 'tier_min', 'int', 'feature',
     'Fastest tier holding any input relation, 0 to 3.'),
    ('query_features', 'tier_max', 'int', 'feature',
     'Slowest tier holding any input relation, 0 to 3.'),
    ('query_features', 'input_mb', 'float', 'feature',
     'Total size of all input relations in MB.'),
    ('query_features', 'query_class', 'string', 'metadata',
     'Prose description of the query shape. For reading, not for training.'),

    # ---- plan feature
    ('plan_features', 'template_id', 'string', 'id',
     'Identifier of one candidate plan.'),
    ('plan_features', 'instance_id', 'string', 'id',
     'The query instance this candidate belongs to. Join key to '
     'query_features.'),
    ('plan_features', 'query_group_id', 'string', 'id',
     'The query group. Use this to split train and test.'),
    ('plan_features', 'plan_strategy', 'string', 'metadata',
     'Prose description of the plan: target tier, partitioning, join '
     'algorithm, pipelining. For reading, not for training.'),
    ('plan_features', 'strategy_tag', 'int', 'metadata',
     'Numeric tag of the strategy that produced this candidate, 1 to 40.'),
    ('plan_features', 'n_ops', 'int', 'feature',
     'Total number of operations in the plan.'),
    ('plan_features', 'n_joins', 'int', 'feature',
     'Number of join operations.'),
    ('plan_features', 'n_sel', 'int', 'feature',
     'Number of selection and subquery-block operations.'),
    ('plan_features', 'n_agg', 'int', 'feature',
     'Number of aggregation operations.'),
    ('plan_features', 'n_sort', 'int', 'feature',
     'Number of sort operations, including the extra sorts a sort-merge '
     'join requires.'),
    ('plan_features', 'join_typ', 'int', 'feature',
     'Join algorithm: 0 nested loop, 1 hash, 2 sort merge, 3 index nested '
     'loop. Categorical, so one-hot encode for a linear model.'),
    ('plan_features', 'access', 'int', 'feature',
     'Access method: 0 sequential, 1 index. Categorical.'),
    ('plan_features', 'n_pipe', 'int', 'feature',
     'Number of operations fused into an in-memory pipeline, so their '
     'output is never written to storage.'),
    ('plan_features', 'pipe_max', 'int', 'feature',
     'Longest pipeline in the plan, in operations.'),
    ('plan_features', 'n_par', 'int', 'feature',
     'Number of operations run on partitioned input.'),
    ('plan_features', 'par_max', 'int', 'feature',
     'Largest partition count actually achieved. Capped by the devices at '
     'the tier, so an operation on l3 cannot exceed 1.'),
    ('plan_features', 'd_0', 'float', 'feature',
     'Volume read and written on tier l0 (HDD) in MB.'),
    ('plan_features', 'd_1', 'float', 'feature',
     'Volume read and written on tier l1 (SSD) in MB.'),
    ('plan_features', 'd_2', 'float', 'feature',
     'Volume read and written on tier l2 (high-speed SSD) in MB.'),
    ('plan_features', 'd_3', 'float', 'feature',
     'Volume read and written on tier l3 (NVMe) in MB.'),
    ('plan_features', 'n_cross', 'int', 'feature',
     'Number of operations whose read tier differs from its write tier.'),
    ('plan_features', 'n_push', 'int', 'feature',
     'Number of operations whose output is written to a faster tier than '
     'the one it was read from.'),
    ('plan_features', 'tier_span', 'int', 'feature',
     'Difference between the slowest and fastest tier the plan touches.'),
    ('plan_features', 'read_total', 'float', 'feature',
     'Total volume read from persistent storage in MB, including re-reads '
     'of intermediate results.'),
    ('plan_features', 'write_total', 'float', 'feature',
     'Total volume written to persistent storage in MB. Lower than the sum '
     'of operation outputs when operations are pipelined.'),
    ('plan_features', 'inter_max', 'float', 'feature',
     'Size of the largest intermediate result in MB. Decides whether an '
     'operation can be pipelined or placed on a small fast tier.'),
    ('plan_features', 'reduce', 'float', 'feature',
     'Final output size divided by total input size. A property of the '
     'query, so constant across the candidates of one instance.'),
    ('plan_features', 'cost', 'float', 'target',
     'Estimated execution cost from the analytical model, in time units. '
     'A regression target; exclude it when predicting which plan wins, or '
     'the task is trivial.'),
    ('plan_features', 'cost_ratio_to_best', 'float', 'target',
     'This candidate cost divided by the best cost of its instance. 1.0 for the winner.'),
    ('plan_features', 'optimal', 'int', 'target',
     '1 if this candidate is the cheapest of its instance, else 0.'),
    ('plan_features', 'plan_template_id', 'string', 'target',
     'template_id of the cheapest candidate of this instance. The label the '
     'framework retrieves.'),
    ('plan_features', 'best_strategy_tag', 'int', 'target',
     'strategy_tag of the cheapest candidate. Multiclass target.'),

    # ---- provenance
    ('instance_provenance', 'instance_id', 'string', 'id',
     'Identifier of the query instance.'),
    ('instance_provenance', 'query_group_id', 'string', 'id',
     'The query group.'),
    ('instance_provenance', 'scale_factor', 'category', 'metadata',
     'Which scale factor produced this instance: SF1, SF3 or SF10. NOT a '
     'feature: a query arriving at runtime carries no such label. Its '
     'effect reaches the model through input_mb and the size classes.'),
    ('instance_provenance', 'placement', 'category', 'metadata',
     'Which storage placement produced this instance. baseline is the '
     'deployed configuration; cold, hot and split are hypothetical '
     'variants standing for the same database after data migration. NOT a '
     'feature: its effect reaches the model through tier_min, tier_max '
     'and the per-tier volumes.'),
]


def write_dictionary(path):
    pd.DataFrame(DICT_ROWS, columns=['file', 'column', 'type', 'role',
                                     'description']).to_csv(path, index=False)


# =====================================================================
# 9. MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Generate the query-plan template dataset.')
    ap.add_argument('outdir', nargs='?',
                    default=os.path.dirname(os.path.abspath(__file__)),
                    help='where to write the files (default: next to this script)')
    ap.add_argument('--no-excel', action='store_true',
                    help='skip the .xlsx file')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    out = lambda n: os.path.join(args.outdir, n)

    qf, pf, prov = generate()

    # ML-ready join. Provenance columns are deliberately left out.
    merged = pf.merge(qf.drop(columns=['query_group_id']),
                      on='instance_id', suffixes=('_plan', '_query'))

    qf.to_csv(out('query_features.csv'), index=False)
    pf.to_csv(out('plan_features.csv'), index=False)
    merged.to_csv(out('template_dataset.csv'), index=False)
    prov.to_csv(out('instance_provenance.csv'), index=False)
    write_dictionary(out('data_dictionary.csv'))

    if not args.no_excel:
        try:
            with pd.ExcelWriter(out('template_dataset.xlsx')) as w:
                qf.to_excel(w, sheet_name='query_features', index=False)
                pf.to_excel(w, sheet_name='plan_features', index=False)
                merged.to_excel(w, sheet_name='ml_ready', index=False)
                prov.to_excel(w, sheet_name='provenance', index=False)
                pd.DataFrame(DICT_ROWS,
                             columns=['file', 'column', 'type', 'role',
                                      'description']).to_excel(
                    w, sheet_name='data_dictionary', index=False)
        except ImportError:
            print('openpyxl not installed, skipping the .xlsx '
                  '(pip install openpyxl)', file=sys.stderr)

    print(f'wrote to {args.outdir}')
    for f in ('query_features.csv', 'plan_features.csv', 'template_dataset.csv',
              'instance_provenance.csv', 'data_dictionary.csv',
              'template_dataset.xlsx'):
        p = out(f)
        if os.path.exists(p):
            print(f'  {f:26s} {os.path.getsize(p):>10,} bytes')

    feats = [r[1] for r in DICT_ROWS
             if r[0] in ('query_features', 'plan_features') and r[3] == 'feature']
    print()
    print(f'query groups     {qf.query_group_id.nunique()}')
    print(f'query instances  {len(qf)}')
    print(f'candidate plans  {len(pf)}')
    print(f'model features   {len(set(feats))}')
    print()
    print('targets: best_strategy_tag (multiclass), optimal (binary), '
          'cost_ratio_to_best and cost (regression)')
    print('split on query_group_id, never at random')


if __name__ == '__main__':
    main()
