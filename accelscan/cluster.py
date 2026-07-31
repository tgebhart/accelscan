"""Stage E2: BERTopic over SPECTER2 embeddings → topic assignments.

Loads the stage-E1 embeddings (all chunks), reloads the aligned abstracts for
c-TF-IDF keywords, runs BERTopic (UMAP → HDBSCAN → c-TF-IDF) on the
*precomputed* embeddings, and writes:
  clusters/{cluster_version}/assignments.parquet   <key>, topic_id, topic_prob
  clusters/{cluster_version}/topics.parquet        topic_id, size, keywords, rep_<key>s
  clusters/{cluster_version}/params.json           embed tag + UMAP/HDBSCAN params

Unsupervised on topic content only — GPU identity is never an input here; it is
joined on later as a dependent overlay (notebooks/gpu_topics.ipynb).

  python -m accelscan.cluster --embed-tag specter2-proximity
  python -m accelscan.cluster --embed-tag specter2-base --local-embed output/embed_smoke \
      --local-out output/clusters_smoke --min-cluster-size 10   # dev
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from accelscan.meta import paper_abstracts
from accelscan.paths import S2ORC, Corpus, clusters_base, embeddings_parts, get_corpus, s3_uri

# UMAP + HDBSCAN defaults (BERTopic recipe); recorded in params.json.
# n_neighbors=30 and min_samples=5 (below min_cluster_size) both lower the
# HDBSCAN noise fraction vs the aggressive defaults (min_samples would
# otherwise equal min_cluster_size). See --reduce-outliers for full coverage.
UMAP_N_NEIGHBORS = 30
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 5
CLUSTER_SELECTION_METHOD = 'eom'   # 'leaf' → more, smaller, lower-noise clusters
CLUSTER_SELECTION_EPSILON = 0.0    # >0 merges near-threshold points (less noise)
VECT_MIN_DF = 10
SEED = 42


def load_embeddings(embed_tag: str, so: dict, local_embed: str | None,
                    corpus: Corpus = S2ORC) -> pl.DataFrame:
    if local_embed:
        src = f'{local_embed}/*.parquet'
        return pl.read_parquet(src).sort(corpus.key)
    src = s3_uri(f'{embeddings_parts(corpus, embed_tag)}/*.parquet')
    return pl.read_parquet(src, storage_options=so).unique(corpus.key).sort(corpus.key)


def load_abstracts_for(ids: list, so: dict, corpus: Corpus = S2ORC) -> pl.DataFrame:
    """Abstracts for the c-TF-IDF topic keywords, via the corpus metadata source."""
    from accelscan.s3 import make_s3_client
    frame = pl.DataFrame({corpus.key: ids}, schema={corpus.key: corpus.key_dtype})
    return paper_abstracts(corpus, frame, so, make_s3_client())


def build_topic_model(min_cluster_size: int, min_samples: int, n_neighbors: int,
                      selection_method: str, selection_epsilon: float):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=UMAP_N_COMPONENTS,
                      min_dist=UMAP_MIN_DIST, metric='cosine', random_state=SEED)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            metric='euclidean', cluster_selection_method=selection_method,
                            cluster_selection_epsilon=selection_epsilon, prediction_data=True)
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=VECT_MIN_DF)
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer, calculate_probabilities=False,
                    verbose=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--embed-tag', default='specter2-base')
    ap.add_argument('--cluster-version', help='output namespace; default = embed-tag + params')
    ap.add_argument('--min-cluster-size', type=int, default=HDBSCAN_MIN_CLUSTER_SIZE)
    ap.add_argument('--min-samples', type=int, default=HDBSCAN_MIN_SAMPLES,
                    help='HDBSCAN conservativeness; lower = less noise')
    ap.add_argument('--n-neighbors', type=int, default=UMAP_N_NEIGHBORS,
                    help='UMAP neighborhood; higher = denser manifold, less noise')
    ap.add_argument('--cluster-selection-method', default=CLUSTER_SELECTION_METHOD,
                    choices=['eom', 'leaf'])
    ap.add_argument('--cluster-selection-epsilon', type=float, default=CLUSTER_SELECTION_EPSILON)
    ap.add_argument('--reduce-outliers', action='store_true',
                    help='reassign noise (-1) papers to nearest topic by embedding cosine')
    ap.add_argument('--local-embed', help='dev: read embeddings from a local dir')
    ap.add_argument('--local-out', help='dev: write clusters to a local dir')
    ap.add_argument('--corpus', default='s2orc', choices=['s2orc', 'arxiv'])
    args = ap.parse_args()
    c = get_corpus(args.corpus)

    from accelscan.s3 import make_s3_client, storage_options
    so = storage_options()

    emb_df = load_embeddings(args.embed_tag, so, args.local_embed, c)
    corpusids = emb_df[c.key].to_list()
    embeddings = np.asarray(emb_df['emb'].to_list(), dtype=np.float32)
    print(f'{len(corpusids):,} embeddings loaded ({embeddings.shape})', file=sys.stderr)

    # align abstracts to embedding order for c-TF-IDF
    abstracts = load_abstracts_for(corpusids, so, c)
    amap = dict(zip(abstracts[c.key].to_list(), abstracts['abstract'].to_list()))
    docs = [amap.get(cid) or '' for cid in corpusids]

    topic_model = build_topic_model(args.min_cluster_size, args.min_samples,
                                    args.n_neighbors, args.cluster_selection_method,
                                    args.cluster_selection_epsilon)
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    topics = [int(t) for t in topics]
    fit_noise = sum(t == -1 for t in topics)

    was_outlier = [False] * len(topics)
    if args.reduce_outliers and fit_noise:
        reduced = topic_model.reduce_outliers(docs, topics, strategy='embeddings',
                                              embeddings=embeddings)
        was_outlier = [orig == -1 and new != -1 for orig, new in zip(topics, reduced)]
        topics = [int(t) for t in reduced]

    assignments = pl.DataFrame({
        c.key: corpusids,
        'topic_id': topics,
        'topic_prob': [float(p) for p in (probs if probs is not None else [0.0] * len(topics))],
        'was_outlier': was_outlier,
    }, schema={c.key: c.key_dtype, 'topic_id': pl.Int32, 'topic_prob': pl.Float32,
               'was_outlier': pl.Boolean})

    # sizes from FINAL assignments (reflects any outlier reassignment)
    sizes = dict(assignments.group_by('topic_id').len().iter_rows())
    all_topics = sorted(sizes)
    rep = {t: assignments.filter(pl.col('topic_id') == t)[c.key].head(5).to_list()
           for t in all_topics}
    topics_df = pl.DataFrame({
        'topic_id': all_topics,
        'size': [sizes[t] for t in all_topics],
        'keywords': [', '.join(w for w, _ in topic_model.get_topic(t)[:10]) if t != -1 else '(noise)'
                     for t in all_topics],
        f'rep_{c.key}s': [rep[t] for t in all_topics],
    }, schema={'topic_id': pl.Int32, 'size': pl.Int64, 'keywords': pl.Utf8,
               f'rep_{c.key}s': pl.List(c.key_dtype)})

    n_topics = sum(t != -1 for t in all_topics)
    noise = int(assignments.filter(pl.col('topic_id') == -1).height)
    params = {'embed_tag': args.embed_tag, 'umap_n_neighbors': args.n_neighbors,
              'umap_n_components': UMAP_N_COMPONENTS, 'umap_min_dist': UMAP_MIN_DIST,
              'hdbscan_min_cluster_size': args.min_cluster_size,
              'hdbscan_min_samples': args.min_samples,
              'cluster_selection_method': args.cluster_selection_method,
              'cluster_selection_epsilon': args.cluster_selection_epsilon,
              'reduce_outliers': bool(args.reduce_outliers), 'seed': SEED,
              'n_papers': len(corpusids), 'n_topics': n_topics,
              'fit_noise_frac': round(fit_noise / max(len(corpusids), 1), 4),
              'final_noise_frac': round(noise / max(len(corpusids), 1), 4),
              'n_reassigned': int(sum(was_outlier))}
    print(f'{n_topics} topics | fit-noise {params["fit_noise_frac"]:.1%} '
          f'| final-noise {params["final_noise_frac"]:.1%}', file=sys.stderr)

    version = (args.cluster_version
               or f'{args.embed_tag}-mcs{args.min_cluster_size}-ms{args.min_samples}'
               + ('-ro' if args.reduce_outliers else ''))
    if args.local_out:
        out = Path(args.local_out); out.mkdir(parents=True, exist_ok=True)
        assignments.write_parquet(out / 'assignments.parquet')
        topics_df.write_parquet(out / 'topics.parquet')
        (out / 'params.json').write_text(json.dumps(params, indent=2))
        print(f'wrote clusters to {out}/', file=sys.stderr)
    else:
        client = make_s3_client()
        base = clusters_base(c, version)
        for name, frame in [('assignments', assignments), ('topics', topics_df)]:
            buf = io.BytesIO(); frame.write_parquet(buf)
            client.put_object(Bucket=BUCKET, Key=f'{base}/{name}.parquet', Body=buf.getvalue())
        client.put_object(Bucket=BUCKET, Key=f'{base}/params.json',
                          Body=json.dumps(params, indent=2).encode())
        print(f'wrote clusters to s3://{BUCKET}/{base}/', file=sys.stderr)


if __name__ == '__main__':
    main()
