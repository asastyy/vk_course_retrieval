#!/usr/bin/env python3
"""Classical text reranking. No ML, pretrained models or external data.

Usage: ./solution.py --submission_file=submission.csv DATA_DIR
The optional cache stores only corpus statistics; labels are read every run.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import snowballstemmer

TOKEN = re.compile(r"(?u)\b\w\w+\b")
STEMMER = snowballstemmer.stemmer('russian')


def tokenize(text):
    return TOKEN.findall(text.lower().replace('ё', 'е'))


@lru_cache(maxsize=1000000)
def stem(word):
    return STEMMER.stemWord(word)


def read_queries(path):
    result = {}
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            qid, text = line.rstrip('\r\n').split('\t', 1)
            result[int(qid)] = text
    return result


def read_qrels(path):
    result = defaultdict(dict)
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            qid, _, did, label = line.split()
            result[int(qid)][did] = int(label)
    return result


def read_docs(path):
    # A streaming reader: never keep the 1 GB of page text in memory.
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            fields = line.rstrip('\r\n').split('\t', 3)
            fields += [''] * (4 - len(fields))
            yield fields


def make_layout(data):
    train = read_queries(data / 'vkmarco-doctrain-queries.tsv')
    test = read_queries(data / 'vkmarco-doceval-queries.tsv')
    qrels = read_qrels(data / 'vkmarco-doctrain-qrels.tsv')
    with open(data / 'objects.csv', encoding='utf-8-sig', newline='') as f:
        objects = [(int(r['ObjectId']), int(r['QueryId']), r['DocumentId'])
                   for r in csv.DictReader(f)]
    if len({r[0] for r in objects}) != len(objects):
        raise ValueError('Duplicate ObjectId')
    test_candidates = defaultdict(list)
    for oid, qid, did in objects:
        test_candidates[qid].append((oid, did))
    groups, records, queries, labels, object_to_row = [], [], [], [], {}
    ntrain = len(qrels)
    for split, mapping in [('train', train), ('test', test)]:
        ids = sorted(qrels if split == 'train' else test_candidates)
        for qid in ids:
            if qid not in mapping:
                raise ValueError(f'Missing query {split}/{qid}')
            qi = len(queries)
            queries.append(tokenize(mapping[qid]))
            start = len(records)
            candidates = ([(None, d) for d in sorted(qrels[qid])]
                          if split == 'train' else sorted(test_candidates[qid]))
            for oid, did in candidates:
                if oid is not None:
                    object_to_row[oid] = len(records)
                records.append((qi, did))
                labels.append(qrels[qid][did] if split == 'train' else -1)
            groups.append((start, len(records)))
    if ntrain < 2:
        raise ValueError('At least two labelled queries are required')
    return queries, records, groups, np.array(labels), objects, object_to_row, ntrain


def phrase_fraction(query, tokens):
    if len(query) < 2:
        return 0.0
    wanted = set(zip(query, query[1:]))
    found = set(zip(tokens, tokens[1:]))
    return len(wanted & found) / len(wanted)


def build_statistics(data, queries, records, groups):
    """Only frequencies of query terms are stored, but lengths/DF use full pages.

    Raw TF-IDF norms use ALL vocabulary terms, calculated in a second stream.
    This avoids the incorrect cosine obtained by restricting document norms to
    query vocabulary. Stemmed DF is calculated on title U body, once per page.
    """
    n = len(records)
    query_stems = [[stem(w) for w in q] for q in queries]
    qs = [sorted(set(q)) for q in query_stems]
    qr = [sorted(set(q)) for q in queries]
    vocab = {t: i for i, t in enumerate(sorted({t for q in qs for t in q}))}
    raw_vocab = {t: i for i, t in enumerate(sorted({t for q in qr for t in q}))}
    offsets = np.zeros(n + 1, dtype=np.int64)
    roffsets = np.zeros(n + 1, dtype=np.int64)
    for i, (qi, _) in enumerate(records):
        offsets[i + 1] = offsets[i] + len(qs[qi])
        roffsets[i + 1] = roffsets[i] + len(qr[qi])
    term_ids = np.empty(offsets[-1], dtype=np.int32)
    raw_ids = np.empty(roffsets[-1], dtype=np.int32)
    qtf = np.empty(offsets[-1], dtype=np.float64)
    raw_qtf = np.empty(roffsets[-1], dtype=np.float64)
    doc_rows = defaultdict(list)
    for i, (qi, did) in enumerate(records):
        s, r = slice(offsets[i], offsets[i+1]), slice(roffsets[i], roffsets[i+1])
        term_ids[s] = [vocab[t] for t in qs[qi]]
        raw_ids[r] = [raw_vocab[t] for t in qr[qi]]
        ct, cr = Counter(query_stems[qi]), Counter(queries[qi])
        qtf[s] = [ct[t] for t in qs[qi]]
        raw_qtf[r] = [cr[t] for t in qr[qi]]
        doc_rows[did].append(i)
    title_tf = np.zeros(len(term_ids))
    body_tf = np.zeros(len(term_ids))
    raw_tf = np.zeros(len(raw_ids))
    raw_title_tf = np.zeros(len(raw_ids))
    lengths = np.zeros((n, 2))
    phrase = np.zeros((n, 2))
    urlmatch = np.zeros(n)
    df = Counter()
    cf = Counter()
    raw_df = Counter()
    totals = np.zeros(2)
    found_docs = set()
    count = 0
    for did, url, title, body in read_docs(data / 'vkmarco-docs.tsv'):
        rt, rb = tokenize(title), tokenize(body)
        raw_counts = Counter(rt + rb)
        raw_df.update(raw_counts.keys())
        # Cache stems by word, not by occurrence.
        st, sb = [stem(t) for t in rt], [stem(t) for t in rb]
        ct, cb = Counter(st), Counter(sb)
        doc_terms = ct.keys() | cb.keys()
        df.update(t for t in doc_terms if t in vocab)
        cf.update({t: ct[t] + cb[t] for t in doc_terms if t in vocab})
        totals += [len(rt), len(rb)]
        count += 1
        if did in doc_rows:
            if did in found_docs:
                raise ValueError(f'Duplicate document {did}')
            found_docs.add(did)
            rt_counts = Counter(rt)
            su = set(stem(t) for t in tokenize(unquote(url)))
            # Build bigrams once for a page even if it has multiple queries.
            bt, bb = set(zip(st, st[1:])), set(zip(sb, sb[1:]))
            for row in doc_rows[did]:
                qi = records[row][0]
                s = slice(offsets[row], offsets[row+1])
                r = slice(roffsets[row], roffsets[row+1])
                title_tf[s] = [ct[t] for t in qs[qi]]
                body_tf[s] = [cb[t] for t in qs[qi]]
                raw_tf[r] = [raw_counts[t] for t in qr[qi]]
                raw_title_tf[r] = [rt_counts[t] for t in qr[qi]]
                lengths[row] = [len(rt), len(rb)]
                bigrams = set(zip(query_stems[qi], query_stems[qi][1:]))
                if bigrams:
                    phrase[row] = [len(bigrams & bt)/len(bigrams), len(bigrams & bb)/len(bigrams)]
                urlmatch[row] = len(set(qs[qi]) & su) / max(1, len(qs[qi]))
        if count % 10000 == 0:
            print(f'Indexed {count} documents', flush=True)
    missing = doc_rows.keys() - found_docs
    if missing:
        raise ValueError(f'{len(missing)} candidate documents missing, e.g. {sorted(missing)[:3]}')
    if count == 0:
        raise ValueError('Empty document corpus')
    raw_idf_map = {t: 1 + math.log((count+1)/(freq+1)) for t, freq in raw_df.items()}
    raw_idf = np.array([raw_idf_map.get(t, 1+math.log(count+1)) for t in raw_vocab])
    norm = np.zeros(n)
    print('Computing full-vocabulary TF-IDF norms', flush=True)
    for did, _, title, body in read_docs(data / 'vkmarco-docs.tsv'):
        if did not in doc_rows:
            continue
        counts = Counter(tokenize(title + ' ' + body))
        value = math.sqrt(sum((v * raw_idf_map[t])**2 for t, v in counts.items()))
        norm[doc_rows[did]] = value
    dfv = np.array([df[t] for t in vocab], dtype=float)
    cfv = np.array([cf[t] for t in vocab], dtype=float)
    return dict(offsets=offsets, roffsets=roffsets, term_ids=term_ids, raw_ids=raw_ids,
                qtf=qtf, raw_qtf=raw_qtf, title_tf=title_tf, body_tf=body_tf,
                raw_tf=raw_tf, raw_title_tf=raw_title_tf, lengths=lengths,
                phrase=phrase, urlmatch=urlmatch, df=dfv, cf=cfv, N=np.array(count),
                average=totals/count, total_length=np.array(totals.sum()),
                raw_idf=raw_idf, raw_norm=norm)


class Formulas:
    def __init__(self, a):
        self.a = a
        self.n = len(a['lengths'])
        self.row = np.repeat(np.arange(self.n), np.diff(a['offsets']))
        self.rrow = np.repeat(np.arange(self.n), np.diff(a['roffsets']))
        self.idf = np.log1p((float(a['N']) - a['df'] + .5)/(a['df']+.5))[a['term_ids']]
        self.pc = np.maximum(a['cf'][a['term_ids']], .5) / max(float(a['total_length']), 1)
        self.lt = a['lengths'][self.row, 0] / max(a['average'][0], 1e-9)
        self.lb = a['lengths'][self.row, 1] / max(a['average'][1], 1e-9)
        self.tf = a['title_tf'] + a['body_tf']

    def add(self, x):
        return np.bincount(self.row, weights=x, minlength=self.n)

    def bm25f(self, k1=1.2, b=.75, wt=3., bt=.3):
        a = self.a
        tw = a['body_tf'] / (1-b+b*self.lb) + wt*a['title_tf']/(1-bt+bt*self.lt)
        return self.add(self.idf * (k1+1)*tw/(k1+tw) * a['qtf'])

    def bm25(self, field, k1=1.2, b=.75):
        if field == 'body':
            tf, length = self.a['body_tf'], self.lb
        elif field == 'title':
            tf, length = self.a['title_tf'], self.lt
        else:
            tf = self.tf
            length = self.a['lengths'][self.row].sum(axis=1) / max(self.a['average'].sum(), 1e-9)
        return self.add(self.idf * (k1+1)*tf/(tf+k1*(1-b+b*length))*self.a['qtf'])

    def lm(self, mu):
        # Query-constant sum log P(t|C) is omitted; rankings are identical.
        length = self.a['lengths'].sum(axis=1)[self.row]
        return self.add((np.log1p(self.tf/(mu*self.pc)) - np.log1p(length/mu))*self.a['qtf'])

    def baseline(self):
        a = self.a
        idf = a['raw_idf'][a['raw_ids']]
        numerator = np.bincount(self.rrow, weights=a['raw_tf']*a['raw_qtf']*idf**2, minlength=self.n)
        qnorm = np.sqrt(np.bincount(self.rrow, weights=(a['raw_qtf']*idf)**2, minlength=self.n))
        return numerator/np.maximum(qnorm*a['raw_norm'], 1e-30)

    def coverage(self, field):
        tf = self.a['title_tf'] if field == 'title' else self.tf
        return self.add(self.idf*(tf > 0))/np.maximum(self.add(self.idf), 1e-30)


class Evaluation:
    def __init__(self, groups, labels, ntrain):
        self.groups = groups
        self.ntrain = ntrain
        width = max(e-s for s, e in groups[:ntrain])
        self.indices = np.zeros((ntrain, width), dtype=np.int64)
        self.mask = np.zeros((ntrain, width), dtype=bool)
        self.gains = np.zeros((ntrain, width))
        for i, (s, e) in enumerate(groups[:ntrain]):
            self.indices[i, :e-s] = np.arange(s, e)
            self.mask[i, :e-s] = True
            self.gains[i, :e-s] = 2.**labels[s:e]-1
        self.k = min(10, width)
        self.discount = 1/np.log2(np.arange(2, self.k+2))
        self.ideal = (np.sort(self.gains, axis=1)[:, ::-1][:, :self.k]*self.discount).sum(axis=1)
        rng = np.random.default_rng(42)
        ids = rng.permutation(ntrain)
        cut = max(1, min(ntrain-1, int(.8*ntrain)))
        self.fit, self.holdout = ids[:cut], ids[cut:]

    def per_query(self, scores):
        scores2d = np.where(self.mask, scores[self.indices], -np.inf)
        # Candidate rows have deterministic order; test rows use ObjectId order.
        order = np.argsort(-scores2d, axis=1, kind='stable')[:, :self.k]
        dcg = (np.take_along_axis(self.gains, order, axis=1)*self.discount).sum(axis=1)
        return np.divide(dcg, self.ideal, out=np.zeros_like(dcg), where=self.ideal > 0)

    def metric(self, scores, subset=None):
        v = self.per_query(scores)
        return float(v.mean() if subset is None else v[subset].mean())

    def normalize(self, scores):
        # Min-max within the query, with zero for a constant signal.
        out = np.zeros_like(scores)
        for s, e in self.groups:
            x = scores[s:e]
            out[s:e] = (x-x.min()) / max(float(np.ptp(x)), 1e-12)
        return out


def select_formula(a, groups, labels, ntrain):
    f, ev = Formulas(a), Evaluation(groups, labels, ntrain)
    experiments = []
    best_value, best_params, best_scores = -1., None, None
    for k1 in [.6, 1.2, 2.0]:
        for b in [.2, .5, .8]:
            for wt in [1., 3., 8.]:
                params = dict(k1=k1, b=b, wt=wt, bt=.3)
                scores = f.bm25f(**params)
                val = ev.metric(scores, ev.fit)
                experiments.append(dict(model='BM25F', parameters=params, fit=val,
                                        holdout=ev.metric(scores, ev.holdout)))
                if val > best_value + 1e-12:
                    best_value, best_params, best_scores = val, params, scores
    features = dict(BM25F=best_scores, TFIDF=f.baseline(),
                    BM25_body=f.bm25('body', b=best_params['b']),
                    BM25_title=f.bm25('title', b=.3),
                    LM_Dirichlet=f.lm(2000),
                    title_coverage=f.coverage('title'),
                    phrase_title=a['phrase'][:, 0], phrase_body=a['phrase'][:, 1],
                    URL_coverage=a['urlmatch'])
    for name, scores in features.items():
        experiments.append(dict(model=name, fit=ev.metric(scores, ev.fit),
                                holdout=ev.metric(scores, ev.holdout), all_train=ev.metric(scores)))
    names = list(features)
    normalized = np.stack([ev.normalize(features[name]) for name in names])
    # A few formula coefficients, tuned by a deterministic discrete coordinate
    # search. No regression, gradient optimization or learning-to-rank library.
    weights = np.zeros(len(names)); weights[0] = 1
    combined = normalized[0].copy()
    best_value = ev.metric(combined, ev.fit)
    for _ in range(2):
        changed = False
        for j in range(1, len(names)):
            base = combined - weights[j]*normalized[j]
            chosen, current = weights[j], best_value
            for weight in [0., .05, .1, .2, .4, .8]:
                value = ev.metric(base+weight*normalized[j], ev.fit)
                if value > current+1e-7:
                    chosen, current = weight, value
            if chosen != weights[j]:
                changed = True
            weights[j] = chosen
            combined = base+chosen*normalized[j]
            best_value = current
        if not changed:
            break
    # Holdout compares only the chosen base formula and the chosen combination.
    # It was not used during parameter search. Its role is explicitly reported.
    use_combination = ev.metric(combined, ev.holdout) > ev.metric(best_scores, ev.holdout)
    final = combined if use_combination else best_scores
    delta = ev.per_query(final)[ev.holdout]-ev.per_query(features['TFIDF'])[ev.holdout]
    se = float(delta.std(ddof=1)/math.sqrt(len(delta))) if len(delta) > 1 else 0.
    report = dict(seed=42, fit_queries=len(ev.fit), holdout_queries=len(ev.holdout),
                  bm25f_parameters=best_params, feature_order=names,
                  combination_weights=weights.tolist(), use_combination=bool(use_combination),
                  baseline_fit=ev.metric(features['TFIDF'], ev.fit),
                  baseline_holdout=ev.metric(features['TFIDF'], ev.holdout),
                  baseline_all_train=ev.metric(features['TFIDF']),
                  final_fit=ev.metric(final, ev.fit), final_holdout=ev.metric(final, ev.holdout),
                  final_all_train=ev.metric(final), holdout_delta=float(delta.mean()),
                  holdout_delta_approx_95ci=[float(delta.mean()-1.96*se), float(delta.mean()+1.96*se)],
                  experiments=experiments)
    return final, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission_file', required=True)
    parser.add_argument('--report_file', default=None)
    parser.add_argument('--cache_dir', default=None, help='optional local statistics cache')
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    layout = make_layout(args.data_dir)
    queries, records, groups, labels, objects, object_to_row, ntrain = layout
    cache = None
    if args.cache_dir:
        # Hash all input bytes, so edits cannot silently reuse stale statistics.
        h = hashlib.sha256(Path(__file__).read_bytes())
        for name in ['vkmarco-docs.tsv', 'vkmarco-doctrain-queries.tsv',
                     'vkmarco-doctrain-qrels.tsv', 'vkmarco-doceval-queries.tsv', 'objects.csv']:
            with open(args.data_dir/name, 'rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    h.update(chunk)
        cache = Path(args.cache_dir) / (h.hexdigest()+'.npz')
        cache.parent.mkdir(parents=True, exist_ok=True)
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as loaded:
            a = {k: loaded[k] for k in loaded.files}
    else:
        a = build_statistics(args.data_dir, queries, records, groups)
        if cache:
            np.savez_compressed(cache, **a)
    scores, report = select_formula(a, groups, labels, ntrain)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite scores')
    target = Path(args.submission_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ObjectId', 'Score'])
        for oid, _, _ in objects:
            writer.writerow([oid, format(float(scores[object_to_row[oid]]), '.12g')])
    report['elapsed_seconds'] = time.monotonic()-start
    report['documents'] = int(a['N'])
    report['submission_rows'] = len(objects)
    report_path = Path(args.report_file) if args.report_file else target.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'experiments'}, ensure_ascii=False, indent=2))
    print(f'Saved {target}', flush=True)


if __name__ == '__main__':
    main()
