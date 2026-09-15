"""Build a trainable derivative of datasets_deepseek_gullibility_v1.

Script equivalent of nb/dedupe.ipynb (the pipeline that produced
datasets_deepseek_gullibility_v0_3.deduped.3992), re-pointed at the v1 raw release.
Written as a script rather than a notebook because v1 is 7.4x larger than v0.3
(29,621 vs 3,992 conversations) and the interactive/plotting parts of the notebook
aren't needed to produce the dataset itself.

Same defects, same fixes (see README "New: the v1 dataset" section):
  1. structural validation           (malformed / non-alternating / too-short)
  2. degenerate & truncated text     (batch generation truncates the final turn)
  3. exact dedup
  4. near-dup collapse (within level, TF-IDF cosine >= NEAR_DUP_THRESHOLD)
  5. cross-level collision removal   (near-identical text, opposite labels)
  6. attribute-leak removal          (gullib/credul/skeptic/naive named outright)
  7. class balancing                 (v1 raw is 57.2% low / 42.8% high)
  8. batch-aware train/holdout split (whole call_index values, never split)
  9. length-matched pair_id          (flagged subset, not filtered -- see below)
 10. write + verify

One deliberate deviation from dedupe.ipynb: step 9 there uses an exact Hungarian
(linear_sum_assignment) 1:1 match, which is O(n^3) and only tractable at v0.3's
scale (~1-2k per class per split). At v1's scale that would be ~10k per class in
the train split alone (10k^3 is not tractable). This script uses a greedy nearest-
neighbour match instead (sklearn NearestNeighbors query + greedy dedup of matches),
which is O(n log n) and gives a slightly looser but still balance-checked pairing.
The shipped corpus (all kept conversations) is unaffected either way -- pair_id is
a supplementary flag for anyone who wants the length-matched subset, not the
primary training view (that's train/ + holdout/, read via .user.txt).

Usage:  python nb/build_v1_deduped.py
"""
from __future__ import annotations

import os
import re
import glob
import json
import math
import time
import shutil
import hashlib
import textwrap
import warnings
import collections
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LEVELS = ["low", "high"]

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.environ.get(
    "V1_RAW_DIR",
    os.path.join(REPO, "datasets_deepseek_gullibility_v1"),
)
SOURCE_DATASET_NAME = "datasets_deepseek_gullibility_v1"

# ---- tunable thresholds -- identical to nb/dedupe.ipynb unless noted ---------- #
NEAR_DUP_THRESHOLD = 0.90
CROSS_LEVEL_THRESHOLD = 0.95
MIN_TURN_PAIRS = 2
MIN_WORDS_PER_TURN = 3
MAX_MATCH_DIST = 1.00
BALANCE_TOL = 0.15
EVAL_FRACTION = 0.20
RANDOM_SEED = 20260915
CLEANING_VERSION = "v1-clean.1"

rng = np.random.default_rng(RANDOM_SEED)

HARD_LEAK_STEMS = ["gullib", "credul", "skeptic", "sceptic", "naive", "naïve"]
SOFT_LEAK_STEMS = ["trust"]
HARD_RE = re.compile("|".join(r"\b" + re.escape(s) for s in HARD_LEAK_STEMS), re.I)
SOFT_RE = re.compile("|".join(r"\b" + re.escape(s) for s in SOFT_LEAK_STEMS), re.I)

TURN_RE = re.compile(r"^(HUMAN|ASSISTANT):\s?(.*)$")
SENTENCE_END = re.compile(r"[.!?…\"')\]]\s*$")
ARTEFACT_RE = re.compile(r"```|^\s*[\{\[]|\bHUMAN:|\bASSISTANT:|<\|.*?\|>")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_transcript(path: str) -> list[tuple[str, str]]:
    out: list[list[str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            if not line.strip():
                continue
            m = TURN_RE.match(line)
            if m:
                out.append(["user" if m.group(1) == "HUMAN" else "assistant",
                            m.group(2).strip()])
            elif out:
                out[-1][1] = (out[-1][1] + " " + line.strip()).strip()
            else:
                out.append(["__untagged__", line.strip()])
    return [(r, c) for r, c in out]


def norm_text(s: str) -> str:
    s = re.sub(r"\b(?:HUMAN|ASSISTANT):\s*", " ", s)
    s = re.sub(r"[^a-z0-9' ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                  / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s else 0.0


def report_balance(frame, tag):
    hi = frame[frame.level == "high"]; lo = frame[frame.level == "low"]
    return {
        "stage": tag, "n": len(frame), "high": len(hi), "low": len(lo),
        "ratio": round(len(lo) / max(1, len(hi)), 3),
        "d_asst_words": round(cohens_d(hi["asst_words"], lo["asst_words"]), 3),
        "d_user_words": round(cohens_d(hi["user_words"], lo["user_words"]), 3),
        "d_total_words": round(cohens_d(hi["total_words"], lo["total_words"]), 3),
    }


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def collapse_near_dups(sub: pd.DataFrame, threshold: float, k: int = 12):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    if len(sub) < 2:
        return sub.index, pd.Index([]), 0
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                          max_features=80000, strip_accents="unicode")
    X = vec.fit_transform(sub["norm_user"] + " " + sub["norm_full"])
    nn = NearestNeighbors(n_neighbors=min(k, len(sub)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    dsu = DSU(len(sub))
    for i in range(len(sub)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if 1 - d >= threshold:
                dsu.union(i, j)
    comp = collections.defaultdict(list)
    for i in range(len(sub)):
        comp[dsu.find(i)].append(i)
    keep_pos, drop_pos = [], []
    for members in comp.values():
        members_sorted = sorted(members, key=lambda p: (-sub["total_words"].iloc[p],
                                                        sub["index"].iloc[p]))
        keep_pos.append(members_sorted[0]); drop_pos.extend(members_sorted[1:])
    multi = sum(1 for m in comp.values() if len(m) > 1)
    return sub.index[keep_pos], sub.index[drop_pos], multi


def greedy_match_pairs(sub: pd.DataFrame, cols, mu, sd, max_dist, balance_tol=BALANCE_TOL):
    """Scalable stand-in for dedupe.ipynb's Hungarian matcher (see module docstring).

    Nearest-neighbour query (high -> low) in standardised length space, then a
    greedy pass over candidates sorted by distance that skips any low index
    already claimed. O(n log n) instead of O(n^3); trades matching optimality
    for tractability at v1's scale. Kept prefix is trimmed to satisfy the same
    |Cohen's d| < balance_tol constraint on every matched covariate.
    """
    from sklearn.neighbors import NearestNeighbors

    hi_df = sub[sub.level == "high"]; lo_df = sub[sub.level == "low"]
    empty = (pd.Index([]), pd.Index([]), {})
    if len(hi_df) == 0 or len(lo_df) == 0:
        return empty

    H = ((hi_df[cols] - mu) / sd).values
    L = ((lo_df[cols] - mu) / sd).values
    k = min(8, len(lo_df))
    nn = NearestNeighbors(n_neighbors=k).fit(L)
    dist, idx = nn.kneighbors(H)

    candidates = []
    for hi_pos in range(len(hi_df)):
        for d, lo_pos in zip(dist[hi_pos], idx[hi_pos]):
            candidates.append((d, hi_pos, lo_pos))
    candidates.sort(key=lambda t: t[0])

    used_hi, used_lo = set(), set()
    pairs = []  # (hi_pos, lo_pos, dist)
    for d, hi_pos, lo_pos in candidates:
        if d > max_dist:
            break
        if hi_pos in used_hi or lo_pos in used_lo:
            continue
        used_hi.add(hi_pos); used_lo.add(lo_pos)
        pairs.append((hi_pos, lo_pos, d))
    if not pairs:
        return empty

    hv = hi_df[cols].values
    lv = lo_df[cols].values
    best_k, best_diag = 0, {}
    n = len(pairs)
    cand_ks = sorted(set(list(range(max(20, n // 50), n + 1, max(1, n // 100))) + [n]))
    for kk in cand_ks:
        hp = [p[0] for p in pairs[:kk]]; lp = [p[1] for p in pairs[:kk]]
        ds = {col: cohens_d(hv[hp, i], lv[lp, i]) for i, col in enumerate(cols)}
        if all(abs(v) < balance_tol for v in ds.values()):
            best_k, best_diag = kk, ds
    if best_k == 0:
        return empty
    kept = pairs[:best_k]
    return (hi_df.index[[p[0] for p in kept]], lo_df.index[[p[1] for p in kept]],
            {"n_candidates": n, "kept": best_k, **best_diag})


def main():
    log(f"repo     : {REPO}")
    log(f"source   : {SRC_DIR}  exists={os.path.isdir(SRC_DIR)}")
    if not os.path.isdir(SRC_DIR):
        raise SystemExit(f"source directory not found: {SRC_DIR}")

    # ---- 0. snapshot -------------------------------------------------------- #
    SNAPSHOT_AT = datetime.now(timezone.utc)
    snapshot_json = sorted(glob.glob(os.path.join(SRC_DIR, "conversation_*.json")))
    SNAPSHOT_STEMS = [os.path.basename(p)[:-5] for p in snapshot_json]
    log(f"conversations in cut: {len(SNAPSHOT_STEMS):,}")

    # ---- 1. load & structural validation ------------------------------------ #
    records = []
    t0 = time.time()
    for stem in SNAPSHOT_STEMS:
        jp = os.path.join(SRC_DIR, stem + ".json")
        tp = os.path.join(SRC_DIR, stem + ".txt")
        rec = {"stem": stem, "drop_reason": None}

        if not os.path.exists(tp):
            rec["drop_reason"] = "missing_transcript"
            records.append(rec)
            continue
        try:
            meta = json.load(open(jp, encoding="utf-8"))
        except Exception as e:
            rec["drop_reason"] = f"unreadable_metadata:{type(e).__name__}"
            records.append(rec)
            continue

        turns = parse_transcript(tp)
        rec.update({
            "level": meta.get("level"), "index": meta.get("index"),
            "topic": meta.get("topic"), "model": meta.get("model"),
            "seed": meta.get("seed"), "call_index": meta.get("call_index"),
            "generated_at": meta.get("generated_at"),
            "orig_leak_stems": tuple(meta.get("leak_stems_found") or ()),
        })

        roles = [r for r, _ in turns]
        user_turns = [c for r, c in turns if r == "user"]
        asst_turns = [c for r, c in turns if r == "assistant"]

        if not turns:
            rec["drop_reason"] = "empty_transcript"
        elif "__untagged__" in roles:
            rec["drop_reason"] = "untagged_line"
        elif rec["level"] not in LEVELS:
            rec["drop_reason"] = f"bad_level:{rec['level']}"
        elif any(roles[i] != ("user" if i % 2 == 0 else "assistant") for i in range(len(roles))):
            rec["drop_reason"] = "non_alternating"
        elif any(not c.strip() for _, c in turns):
            rec["drop_reason"] = "empty_turn"
        elif len(user_turns) < MIN_TURN_PAIRS + 1 or len(asst_turns) < MIN_TURN_PAIRS:
            rec["drop_reason"] = f"too_few_exchanges:{len(user_turns)}u/{len(asst_turns)}a"

        rec.update({
            "turns": turns,
            "n_turns": len(turns), "n_user": len(user_turns), "n_asst": len(asst_turns),
            "user_text": " ".join(user_turns), "asst_text": " ".join(asst_turns),
            "user_words": sum(len(c.split()) for c in user_turns),
            "asst_words": sum(len(c.split()) for c in asst_turns),
            "min_turn_words": min((len(c.split()) for _, c in turns), default=0),
            "last_turn": turns[-1][1] if turns else "",
            "norm_full": norm_text(" ".join(c for _, c in turns)),
            "norm_user": norm_text(" ".join(user_turns)),
        })
        records.append(rec)

    df = pd.DataFrame(records)
    df["total_words"] = df["user_words"].fillna(0) + df["asst_words"].fillna(0)
    log(f"parsed {len(df):,} conversations in {time.time() - t0:.1f}s")

    stage1 = df["drop_reason"].notna()
    log(f"STAGE 1 structural  : dropped {int(stage1.sum()):,}")
    if stage1.any():
        print(df.loc[stage1, "drop_reason"].value_counts().to_string())

    # ---- 2. degenerate & truncated text -------------------------------------- #
    def flag_degenerate(row):
        if row["drop_reason"]:
            return row["drop_reason"]
        if not SENTENCE_END.search(row["last_turn"] or ""):
            return "truncated_final_turn"
        if (row["min_turn_words"] or 0) < MIN_WORDS_PER_TURN:
            return "stub_turn"
        body = " ".join(c for _, c in row["turns"])
        if ARTEFACT_RE.search(body):
            return "text_artefact"
        if len(set(row["norm_user"].split())) < 10:
            return "degenerate_user_text"
        return None

    df["drop_reason"] = df.apply(flag_degenerate, axis=1)
    stage2 = df["drop_reason"].notna() & ~stage1
    log(f"STAGE 2 degenerate  : dropped {int(stage2.sum()):,}; {int(df['drop_reason'].isna().sum()):,} alive")
    if stage2.any():
        print(df.loc[stage2, "drop_reason"].value_counts().to_string())

    # ---- 3. exact duplicates -------------------------------------------------- #
    alive = df["drop_reason"].isna()
    work = df[alive].copy()
    work["full_hash"] = work["norm_full"].map(lambda s: hashlib.sha1(s.encode()).hexdigest())
    work["user_hash"] = work["norm_user"].map(lambda s: hashlib.sha1(s.encode()).hexdigest())

    work = work.sort_values(["full_hash", "index"])
    dup_full = work.duplicated("full_hash", keep="first")
    df.loc[work.index[dup_full], "drop_reason"] = "exact_duplicate"

    work = work[~dup_full].sort_values(["user_hash", "index"])
    dup_user = work.duplicated("user_hash", keep="first")
    df.loc[work.index[dup_user], "drop_reason"] = "duplicate_user_text"
    work = work[~dup_user]

    stage3 = df["drop_reason"].isin(["exact_duplicate", "duplicate_user_text"])
    log(f"STAGE 3 exact dup   : dropped {int(stage3.sum()):,}; {len(work):,} alive")

    # ---- 4. near-duplicate collapse (within level) --------------------------- #
    log(f"STAGE 4 near-dup collapse at cosine >= {NEAR_DUP_THRESHOLD} (this can take a minute)")
    for lv in LEVELS:
        t1 = time.time()
        sub = work[work.level == lv]
        keep_i, drop_i, n_clusters = collapse_near_dups(sub, NEAR_DUP_THRESHOLD)
        df.loc[drop_i, "drop_reason"] = "near_duplicate"
        log(f"  {lv:<5} before={len(sub):,} clusters_collapsed={n_clusters:,} "
            f"dropped={len(drop_i):,} after={len(keep_i):,}  ({time.time()-t1:.1f}s)")
    work = df[df["drop_reason"].isna()].copy()
    log(f"  {len(work):,} alive")

    # ---- 5. cross-level collisions -------------------------------------------- #
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    t1 = time.time()
    vec_x = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                            max_features=80000, strip_accents="unicode")
    Xx = vec_x.fit_transform(work["norm_user"])
    nnx = NearestNeighbors(n_neighbors=min(6, len(work)), metric="cosine").fit(Xx)
    dx, ix = nnx.kneighbors(Xx)
    levels_arr = work["level"].values

    collisions = set()
    for i in range(len(work)):
        for d, j in zip(dx[i, 1:], ix[i, 1:]):
            if (1 - d) >= CROSS_LEVEL_THRESHOLD and levels_arr[i] != levels_arr[j]:
                collisions.add(i); collisions.add(j)

    coll_idx = work.index[sorted(collisions)]
    df.loc[coll_idx, "drop_reason"] = "cross_level_collision"
    log(f"STAGE 5 cross-level : dropped {len(coll_idx):,} label-ambiguous  ({time.time()-t1:.1f}s)")
    work = df[df["drop_reason"].isna()].copy()
    log(f"  {len(work):,} alive")

    # ---- 6. attribute-leak removal -------------------------------------------- #
    work["hard_leak_user"] = work["user_text"].map(lambda t: bool(HARD_RE.search(t)))
    work["hard_leak_asst"] = work["asst_text"].map(lambda t: bool(HARD_RE.search(t)))
    hard = work["hard_leak_user"] | work["hard_leak_asst"]
    df.loc[work.index[hard], "drop_reason"] = "attribute_leak"
    work = df[df["drop_reason"].isna()].copy()
    work["soft_leak"] = (work["user_text"] + " " + work["asst_text"]).map(
        lambda t: bool(SOFT_RE.search(t)))
    log(f"STAGE 6 leakage     : dropped {int(hard.sum()):,} hard leaks; "
        f"{int(work['soft_leak'].sum()):,} soft ('trust') kept+flagged; {len(work):,} alive")

    # ---- 7. batch-aware train/eval split -------------------------------------- #
    call_counts = (
        work.pivot_table(index="call_index", columns="level", values="stem",
                         aggfunc="count", observed=False).fillna(0).astype(int)
    )
    for lv in LEVELS:
        if lv not in call_counts:
            call_counts[lv] = 0
    call_counts["total"] = call_counts[LEVELS].sum(axis=1)
    call_counts["pairable"] = call_counts[LEVELS].min(axis=1) * 2

    target = EVAL_FRACTION * float(call_counts["pairable"].sum())
    order = list(call_counts.sample(frac=1, random_state=RANDOM_SEED).index)
    eval_calls, acc = [], 0.0
    for c in order:
        if acc >= target:
            break
        row = call_counts.loc[c]
        if row["pairable"] == 0:
            continue
        size = float(row["pairable"])
        if acc and acc + size > target * 1.15:
            continue
        eval_calls.append(c); acc += size

    work["split"] = np.where(work["call_index"].isin(eval_calls), "eval", "train")
    log(f"STAGE 7 split       : {len(eval_calls)} calls -> eval "
        f"({float((work.split == 'eval').mean()):.1%} of conversations)")

    # ---- 8a. class balance within each split ---------------------------------- #
    LEN_COLS = ["user_words", "asst_words"]
    mu = work[LEN_COLS].mean()
    sd = work[LEN_COLS].std().replace(0, 1)
    before_bal = report_balance(work, "before balancing")

    keep_bal = []
    for sp in ["train", "eval"]:
        sub = work[work.split == sp]
        hi_i = sub.index[sub.level == "high"]; lo_i = sub.index[sub.level == "low"]
        take = min(len(hi_i), len(lo_i))
        keep_bal += list(rng.choice(hi_i, take, replace=False))
        keep_bal += list(rng.choice(lo_i, take, replace=False))
    n_surplus = len(work.index.difference(pd.Index(keep_bal)))
    df.loc[work.index.difference(pd.Index(keep_bal)), "drop_reason"] = "class_balance_surplus"
    work = df[df["drop_reason"].isna()].copy()
    work["split"] = np.where(work["call_index"].isin(eval_calls), "eval", "train")
    log(f"STAGE 8a balance    : dropped {n_surplus:,} surplus; {len(work):,} alive")

    # ---- 8b. length-matched pair_id (flagged, not filtered) ------------------- #
    pair_of = {}
    next_pair_id = 0
    for sp in ["train", "eval"]:
        sub = work[work.split == sp]
        k_hi, k_lo, diag = greedy_match_pairs(sub, LEN_COLS, mu, sd, MAX_MATCH_DIST)
        for ih, il in zip(k_hi, k_lo):
            pair_of[ih] = next_pair_id; pair_of[il] = next_pair_id; next_pair_id += 1
        log(f"STAGE 8b {sp:<5}   : {diag.get('kept', 0):,} length-matched pairs "
            f"(of {diag.get('n_candidates', 0):,} candidates)")

    work["pair_id"] = work.index.map(pair_of)
    work["length_matched"] = work["pair_id"].notna()
    work["soft_leak"] = (work["user_text"] + " " + work["asst_text"]).map(
        lambda t: bool(SOFT_RE.search(t)))

    after_bal = report_balance(work, "after balancing (all kept)")
    matched_bal = report_balance(work[work.length_matched], "length-matched subset")
    log("BALANCE & LENGTH CONFOUND")
    print(pd.DataFrame([before_bal, after_bal, matched_bal]).to_string(index=False))

    split_tab = (
        work.pivot_table(index="split", columns="level", values="stem",
                         aggfunc="count", observed=False).fillna(0).astype(int)
    )
    split_tab["total"] = split_tab.sum(axis=1)
    log("FINAL SPLIT")
    print(split_tab.to_string())
    overlap = len(set(work.loc[work.split == "train", "call_index"]) &
                  set(work.loc[work.split == "eval", "call_index"]))
    log(f"  calls shared between train/eval: {overlap}  (must be 0)")

    # ---- 9. write the dataset -------------------------------------------------- #
    N_FINAL = len(work)
    OUT_DIR = os.path.join(REPO, f"datasets_deepseek_gullibility_v1.deduped.{N_FINAL}")
    log(f"writing to {OUT_DIR}")
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    SPLIT_DIR = {"train": "train", "eval": "holdout"}
    for dirname in SPLIT_DIR.values():
        os.makedirs(os.path.join(OUT_DIR, dirname), exist_ok=True)

    work = work.sort_values(["level", "call_index", "index"]).reset_index(drop=True)
    counters = {lv: 0 for lv in LEVELS}
    manifest_rows = {"train": [], "holdout": []}

    for _, r in work.iterrows():
        lv = r["level"]
        new_idx = counters[lv]; counters[lv] += 1
        new_stem = f"conversation_{new_idx}_gullibility_{lv}"
        dirname = SPLIT_DIR[r["split"]]
        conv_dir = os.path.join(OUT_DIR, dirname)

        full_txt = "\n".join(
            ("HUMAN: " if role == "user" else "ASSISTANT: ") + content
            for role, content in r["turns"]
        )
        user_txt = "\n".join(content for role, content in r["turns"] if role == "user")

        with open(os.path.join(conv_dir, new_stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(full_txt + "\n")
        with open(os.path.join(conv_dir, new_stem + ".user.txt"), "w", encoding="utf-8") as fh:
            fh.write(user_txt + "\n")

        meta = {
            "attribute": "gullibility",
            "level": lv,
            "topic": r["topic"],
            "index": new_idx,
            "model": r["model"],
            "seed": r["seed"],
            "call_index": int(r["call_index"]) if pd.notna(r["call_index"]) else None,
            "generated_at": r["generated_at"],
            "num_turns": int(r["n_turns"]),
            "n_user_turns": int(r["n_user"]),
            "n_assistant_turns": int(r["n_asst"]),
            "user_words": int(r["user_words"]),
            "assistant_words": int(r["asst_words"]),
            "split": r["split"],
            "pair_id": int(r["pair_id"]) if pd.notna(r.get("pair_id")) else None,
            "length_matched": bool(r["length_matched"]),
            "soft_leak_trust": bool(r["soft_leak"]),
            "leak_stems_found": [],
            "source_stem": r["stem"],
            "source_dataset": SOURCE_DATASET_NAME,
            "snapshot_at": SNAPSHOT_AT.isoformat(),
            "cleaning_version": CLEANING_VERSION,
        }
        with open(os.path.join(conv_dir, new_stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        manifest_rows[dirname].append({
            "stem": new_stem, "level": lv, "index": new_idx, "split": r["split"],
            "pair_id": r.get("pair_id"), "length_matched": bool(r["length_matched"]),
            "topic": r["topic"], "call_index": r["call_index"], "n_turns": r["n_turns"],
            "user_words": r["user_words"], "asst_words": r["asst_words"],
            "soft_leak_trust": bool(r["soft_leak"]), "source_stem": r["stem"],
        })

    all_rows = manifest_rows["train"] + manifest_rows["holdout"]
    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
    for dirname in ("train", "holdout"):
        pd.DataFrame(manifest_rows[dirname]).to_csv(
            os.path.join(OUT_DIR, dirname, "manifest.csv"), index=False)

    dropped = df[df["drop_reason"].notna()][["stem", "level", "index", "call_index", "drop_reason"]]
    dropped.to_csv(os.path.join(OUT_DIR, "dropped.csv"), index=False)

    with open(os.path.join(OUT_DIR, "splits.json"), "w") as fh:
        json.dump({
            "eval_calls": sorted(int(c) for c in eval_calls),
            "train_calls": sorted(int(c) for c in call_counts.index if c not in eval_calls),
            "eval_fraction_target": EVAL_FRACTION,
            "eval_fraction_actual": round(float((work.split == "eval").mean()), 4),
            "holdout_calls": sorted(int(c) for c in eval_calls),
            "holdout_fraction_target": EVAL_FRACTION,
            "holdout_fraction_actual": round(float((work.split == "eval").mean()), 4),
            "folder_layout": {"train": "train/", "holdout": "holdout/"},
            "random_seed": RANDOM_SEED,
        }, fh, indent=2)

    n_train, n_holdout = len(manifest_rows["train"]), len(manifest_rows["holdout"])
    log(f"wrote {N_FINAL:,} conversations ({counters['high']:,} high / {counters['low']:,} low)")
    log(f"  train/   : {n_train:,} conversations")
    log(f"  holdout/ : {n_holdout:,} conversations")

    # ---- 10. verification ------------------------------------------------------ #
    out_paths = sorted(
        (dirname, os.path.basename(p)[:-5])
        for dirname in ("train", "holdout")
        for p in glob.glob(os.path.join(OUT_DIR, dirname, "conversation_*.json"))
    )
    v_rows = []
    for dirname, stem in out_paths:
        conv_dir = os.path.join(OUT_DIR, dirname)
        meta = json.load(open(os.path.join(conv_dir, stem + ".json")))
        full = open(os.path.join(conv_dir, stem + ".txt"), encoding="utf-8").read()
        turns = parse_transcript(os.path.join(conv_dir, stem + ".txt"))
        v_rows.append({
            "stem": f"{dirname}/{stem}", "folder": dirname,
            "level": meta["level"], "split": meta["split"], "call_index": meta["call_index"],
            "user_text": " ".join(c for r, c in turns if r == "user"),
            "asst_text": " ".join(c for r, c in turns if r == "assistant"),
            "norm_full": norm_text(full),
            "norm_user": norm_text(" ".join(c for r, c in turns if r == "user")),
        })
    out = pd.DataFrame(v_rows)
    log(f"reloaded {len(out):,} conversations from disk for verification")

    checks = []
    def chk(name, passed, detail):
        checks.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    chk("every .json has .txt and .user.txt",
        all(os.path.exists(os.path.join(OUT_DIR, dirname, stem + e))
            for dirname, stem in out_paths for e in (".txt", ".user.txt")),
        f"{len(out_paths):,} triples")
    chk("folder assignment matches split field",
        (out["folder"].map({"train": "train", "holdout": "eval"}) == out["split"]).all(),
        f"{len(out):,} conversations")
    chk("no exact duplicate conversations", int(out["norm_full"].duplicated().sum()) == 0,
        f"{int(out['norm_full'].duplicated().sum())} dupes")
    chk("no duplicate user-side text", int(out["norm_user"].duplicated().sum()) == 0,
        f"{int(out['norm_user'].duplicated().sum())} dupes")
    chk("classes balanced", out["level"].value_counts().nunique() == 1,
        out["level"].value_counts().to_dict())
    chk("no hard attribute leak",
        not (out["user_text"] + out["asst_text"]).str.contains(HARD_RE).any(),
        f"{int((out['user_text'] + out['asst_text']).str.contains(HARD_RE).sum())} hits")
    chk("train/eval share no call_index",
        len(set(out.loc[out.split == "train", "call_index"]) &
            set(out.loc[out.split == "eval", "call_index"])) == 0,
        f"eval calls: {sorted(set(out.loc[out.split=='eval','call_index']))[:5]}...")

    log("VERIFICATION")
    print(pd.DataFrame(checks).to_string(index=False))
    any_fail = any(c["result"] == "FAIL" for c in checks)

    # ---- 11. cleaning report + README ------------------------------------------ #
    DROP_STAGE = {
        "missing_transcript": "1 structural", "unreadable_metadata": "1 structural",
        "empty_transcript": "1 structural", "untagged_line": "1 structural",
        "non_alternating": "1 structural", "empty_turn": "1 structural",
        "bad_level": "1 structural", "too_few_exchanges": "1 structural",
        "truncated_final_turn": "2 degenerate", "stub_turn": "2 degenerate",
        "text_artefact": "2 degenerate", "degenerate_user_text": "2 degenerate",
        "exact_duplicate": "3 exact dup", "duplicate_user_text": "3 exact dup",
        "near_duplicate": "4 near dup", "cross_level_collision": "5 label noise",
        "attribute_leak": "6 leakage", "class_balance_surplus": "7 balancing",
    }
    reason = df["drop_reason"].fillna("KEPT").map(
        lambda r: r.split(":")[0] if isinstance(r, str) else r)
    ledger = reason.value_counts().rename("conversations").to_frame()
    ledger["stage"] = ledger.index.map(lambda r: DROP_STAGE.get(r, "-- kept --"))
    ledger["pct_of_source"] = (100 * ledger["conversations"] / len(df)).round(2)
    ledger = ledger.sort_values(["stage", "conversations"], ascending=[True, False])

    R = []
    A = R.append
    A("=" * 86)
    A(f"  CLEANING REPORT -- {os.path.basename(OUT_DIR)}")
    A("=" * 86)
    A(f"  built at            : {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    A(f"  source dataset      : {SOURCE_DATASET_NAME}")
    A(f"  source snapshot at  : {SNAPSHOT_AT:%Y-%m-%d %H:%M:%S} UTC")
    A(f"  cleaning version    : {CLEANING_VERSION}   (seed {RANDOM_SEED})")
    A(f"  pairing method      : greedy nearest-neighbour (not exact Hungarian -- see module docstring)")
    A("")
    A(f"  source conversations: {len(df):,}")
    A(f"  kept                : {len(out):,}  ({len(out)/len(df):.1%})")
    A(f"  dropped             : {len(df) - len(out):,}  ({1 - len(out)/len(df):.1%})")
    A("")
    A("-" * 86)
    A("  COMPOSITION")
    A("-" * 86)
    A(f"  high / low          : {int((out.level=='high').sum()):,} / {int((out.level=='low').sum()):,}")
    A(f"  train / holdout     : {int((out.split=='train').sum()):,} / {int((out.split=='eval').sum()):,}")
    A(f"  length-matched pairs: {int(work['length_matched'].sum()) // 2:,}")
    A("")
    A("-" * 86)
    A("  ATTRITION")
    A("-" * 86)
    for r in ledger.index:
        if r == "KEPT":
            continue
        A(f"  {DROP_STAGE.get(r, '?'):<16} {r:<24} {ledger.loc[r, 'conversations']:>6,}"
          f"  ({ledger.loc[r, 'pct_of_source']:>5.2f}%)")
    A("")
    A("-" * 86)
    A("  BALANCE / LENGTH CONFOUND (before -> after)")
    A("-" * 86)
    A(f"  class ratio low:high    : {before_bal['ratio']} -> {after_bal['ratio']}")
    A(f"  user length d           : {before_bal['d_user_words']} -> {after_bal['d_user_words']}")
    A(f"  assistant length d      : {before_bal['d_asst_words']} -> {after_bal['d_asst_words']}"
      f"   (kept as-is; see pair_id subset, d={matched_bal['d_asst_words']})")
    A("=" * 86)
    report = "\n".join(R)
    with open(os.path.join(OUT_DIR, "cleaning_report.txt"), "w") as fh:
        fh.write(report)
    print(report)

    readme = f"""# datasets_deepseek_gullibility_v1.deduped.{N_FINAL}

Training-ready derivative of `datasets_deepseek_gullibility_v1`, built by
`nb/build_v1_deduped.py` -- the same pipeline as `nb/dedupe.ipynb` (which produced
`datasets_deepseek_gullibility_v0_3.deduped.3992`), re-pointed at the v1 raw release.
See that script's module docstring for the one deliberate deviation (greedy nearest-
neighbour length pairing instead of exact Hungarian assignment, for tractability at
7.4x the scale).

* **{N_FINAL:,} conversations** -- {int((out.level=='high').sum()):,} high / {int((out.level=='low').sum()):,} low (exactly balanced)
* built from a snapshot of the source taken at **{SNAPSHOT_AT:%Y-%m-%d %H:%M:%S} UTC**
* cleaning version `{CLEANING_VERSION}`, seed `{RANDOM_SEED}`

## Layout

```
datasets_deepseek_gullibility_v1.deduped.{N_FINAL}/
  train/                     {int((out.split=='train').sum()):,} conversations
    conversation_<i>_gullibility_<level>.txt
    conversation_<i>_gullibility_<level>.user.txt
    conversation_<i>_gullibility_<level>.json
    manifest.csv
  holdout/                   {int((out.split=='eval').sum()):,} conversations
    conversation_<i>_gullibility_<level>.txt
    conversation_<i>_gullibility_<level>.user.txt
    conversation_<i>_gullibility_<level>.json
    manifest.csv
  manifest.csv                combined manifest (all {N_FINAL:,} rows, with a `split` column)
  dropped.csv                  every dropped conversation and why
  splits.json                  which call_index values went to holdout
  cleaning_report.txt
```

## What was removed

{chr(10).join(f'* `{r}` -- {ledger.loc[r, "conversations"]:,}' for r in ledger.index if r != 'KEPT')}

## How to use it

* **Read training data from `train/`, evaluation data from `holdout/`** -- the split is
  by whole `call_index` values, not random, to avoid batch fingerprint leakage.
* **Prefer `.user.txt`** for extracting activations or training a probe (same rationale
  as v0.3 -- see the top-level README).
* `soft_leak_trust` marks conversations containing the word "trust"; exclude them if you
  want a lexically pristine eval set.

Regenerate with `python nb/build_v1_deduped.py` (set `V1_RAW_DIR` to point at a
different extraction of the raw v1 release if needed).
"""
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)

    log(f"DONE -- {OUT_DIR}")
    if any_fail:
        raise SystemExit("verification FAILED -- see checks above")


if __name__ == "__main__":
    main()
