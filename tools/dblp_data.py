#!/usr/bin/env python3
# dblp_data.py
#
# Build DBLP HIN CSR matrices for SpGEMM benchmarks.
# Matrices (CSR32):
#   PA: P x A (paper -> author)
#   PT: P x T (paper -> term)
#   PC: P x C (paper -> venue)
# And transposes:
#   AP, TP, CP
#
# Output files (binary):
#   <XX>_row_offsets.i32, <XX>_col_indices.i32, <XX>_values.f32
# plus meta.json
#
# Key fix: use lxml + DTD entity resolution (DBLP contains entities like &uuml;).
#
# Requirements:
#   pip install lxml numpy

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from typing import List, Tuple, Dict

DBLP_BASE = "https://dblp.org/xml"
DBLP_XML_GZ_URL = f"{DBLP_BASE}/dblp.xml.gz"
DBLP_DTD_URL    = f"{DBLP_BASE}/dblp.dtd"
DBLP_MD5_URL    = f"{DBLP_BASE}/dblp.xml.gz.md5"

PUB_TAGS = {"article", "inproceedings"}

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","in","into","is","it",
    "its","of","on","or","that","the","their","to","was","were","will","with","without",
    "we","our","you","your","they","them","this","these","those","there","here",
    "using","use","used","via","based","new","towards","toward","over","under",
}
TOKEN_RE = re.compile(r"[a-z]{3,}")  # >=3 letters

def _mkdir_p(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _http_download(url: str, out_path: str, chunk: int = 1 << 20) -> None:
    tmp = out_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
    os.replace(tmp, out_path)

def _read_remote_md5(md5_url: str) -> str:
    req = urllib.request.Request(md5_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        text = r.read().decode("utf-8", errors="ignore").strip()
    return text.split()[0]

def _md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _decompress_gz(gz_path: str, xml_path: str) -> None:
    tmp = xml_path + ".part"
    with gzip.open(gz_path, "rb") as fin, open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    os.replace(tmp, xml_path)

def _tokenize_title(title: str, max_terms_per_paper: int) -> List[str]:
    if not title:
        return []
    s = title.lower()
    toks = [t for t in TOKEN_RE.findall(s) if t not in STOPWORDS]
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_terms_per_paper:
            break
    return out

def _venue_of(elem) -> str:
    bt = elem.findtext("booktitle")
    if bt:
        return bt.strip()
    jr = elem.findtext("journal")
    if jr:
        return jr.strip()
    return ""

def _year_of(elem) -> int:
    y = elem.findtext("year")
    if not y:
        return -1
    try:
        return int(y.strip())
    except Exception:
        return -1

def _authors_of(elem, max_authors_per_paper: int) -> List[str]:
    authors = []
    for a in elem.findall("author"):
        if a.text:
            name = a.text.strip()
            if name:
                authors.append(name)
        if len(authors) >= max_authors_per_paper:
            break
    out, seen = [], set()
    for x in authors:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def coo_to_csr_numpy(n_row: int, n_col: int, rows: List[int], cols: List[int]) -> Tuple[List[int], List[int], List[float]]:
    import numpy as np
    if len(rows) == 0:
        return [0]*(n_row+1), [], []
    r = np.asarray(rows, dtype=np.int32)
    c = np.asarray(cols, dtype=np.int32)
    order = np.lexsort((c, r))
    r = r[order]
    c = c[order]
    if r.size > 1:
        same = (r[1:] == r[:-1]) & (c[1:] == c[:-1])
        keep = np.ones(r.size, dtype=bool)
        keep[1:] = ~same
        r = r[keep]
        c = c[keep]
    counts = np.bincount(r, minlength=n_row).astype(np.int64)
    row_offsets = np.zeros(n_row + 1, dtype=np.int64)
    row_offsets[1:] = np.cumsum(counts)
    col_indices = c.astype(np.int32)
    values = np.ones(col_indices.size, dtype=np.float32)
    return row_offsets.astype(np.int32).tolist(), col_indices.tolist(), values.tolist()

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", default="dblp", help="output directory")
    ap.add_argument("--year_min", type=int, default=2015)
    ap.add_argument("--year_max", type=int, default=2024)

    ap.add_argument("--max_papers", type=int, default=2000000)

    ap.add_argument("--max_authors", type=int, default=800000)
    ap.add_argument("--max_terms", type=int, default=20000)
    ap.add_argument("--max_venues", type=int, default=80000)

    ap.add_argument("--max_authors_per_paper", type=int, default=8)
    ap.add_argument("--max_terms_per_paper", type=int, default=12)

    ap.add_argument("--author_max_papers", type=int, default=600)
    ap.add_argument("--venue_max_papers", type=int, default=150)

    ap.add_argument("--author_min_df", type=int, default=2)
    ap.add_argument("--venue_min_df", type=int, default=5)

    ap.add_argument("--term_min_df", type=int, default=30)
    ap.add_argument("--term_max_df", type=int, default=300)

    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_decompress", action="store_true")
    ap.add_argument("--dblp_xml_gz", default="", help="use local dblp.xml.gz if provided")
    ap.add_argument("--dblp_xml", default="", help="use local dblp.xml if provided")

    ap.add_argument("--resume", action="store_true",
                    help="if papers.jsonl.gz exists, skip PARSE1 and reuse it")

    ap.add_argument("--compact_ids", action="store_true")

    # estimate APCPA upper-bound
    ap.add_argument("--estimate_apcpa", action="store_true",
                    help="estimate APCPA nnz upper-bound ~ sum_v |A_v|^2 (after pass2)")
    ap.add_argument("--max_apcpa_est", type=int, default=600000000,
                    help="warn if estimate exceeds this threshold")

    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    _mkdir_p(out_dir)

    gz_path  = os.path.abspath(args.dblp_xml_gz) if args.dblp_xml_gz else os.path.join(out_dir, "dblp.xml.gz")
    dtd_path = os.path.join(out_dir, "dblp.dtd")
    xml_path = os.path.abspath(args.dblp_xml) if args.dblp_xml else os.path.join(out_dir, "dblp.xml")

    papers_jsonl_gz = os.path.join(out_dir, "papers.jsonl.gz")
    stats_json      = os.path.join(out_dir, "stats_pass1.json")

    # ---- download ----
    if not args.skip_download and (not args.dblp_xml and not args.dblp_xml_gz):
        if not os.path.exists(gz_path):
            print(f"[DL] {DBLP_XML_GZ_URL} -> {gz_path}")
            _http_download(DBLP_XML_GZ_URL, gz_path)
        if not os.path.exists(dtd_path):
            print(f"[DL] {DBLP_DTD_URL} -> {dtd_path}")
            _http_download(DBLP_DTD_URL, dtd_path)
        try:
            remote_md5 = _read_remote_md5(DBLP_MD5_URL)
            local_md5  = _md5_file(gz_path)
            if remote_md5 == local_md5:
                print("[OK] MD5 verified for dblp.xml.gz")
            else:
                print(f"[WARN] MD5 mismatch: remote={remote_md5} local={local_md5}")
        except Exception as e:
            print(f"[WARN] skip md5 verify: {e}")
    else:
        if not os.path.exists(dtd_path) and not args.skip_download:
            print(f"[DL] {DBLP_DTD_URL} -> {dtd_path}")
            _http_download(DBLP_DTD_URL, dtd_path)

    # ---- decompress ----
    if (not args.skip_decompress) and (not args.dblp_xml) and (not os.path.exists(xml_path)):
        if not os.path.exists(gz_path):
            print(f"ERROR: missing {gz_path}")
            sys.exit(1)
        print(f"[GZ] Decompress {gz_path} -> {xml_path}")
        _decompress_gz(gz_path, xml_path)

    # Ensure DTD next to XML (critical for resolving &uuml; etc)
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    dtd_in_xml_dir = os.path.join(xml_dir, "dblp.dtd")
    if os.path.exists(dtd_path) and (not os.path.exists(dtd_in_xml_dir)):
        try:
            shutil.copy2(dtd_path, dtd_in_xml_dir)
        except Exception:
            pass

    # ---- lxml import ----
    try:
        from lxml import etree
    except Exception:
        print("ERROR: lxml is required. Please run: pip install lxml")
        sys.exit(1)

    author_freq = collections.Counter()
    venue_freq  = collections.Counter()
    term_df     = collections.Counter()

    selected = 0
    do_parse1 = True
    if args.resume and os.path.exists(papers_jsonl_gz):
        do_parse1 = False
        print(f"[RESUME] found {papers_jsonl_gz}, skip PARSE1")

    # ---- pass1: parse DBLP XML and write papers.jsonl.gz ----
    if do_parse1:
        if not os.path.exists(xml_path):
            print(f"ERROR: missing {xml_path}. If you only have dblp.xml.gz, do not pass --skip_decompress.")
            sys.exit(1)

        print(f"[PARSE1] iterparse {xml_path} (year {args.year_min}-{args.year_max}) ...")
        orig_cwd = os.getcwd()
        try:
            os.chdir(xml_dir)
            xml_rel = os.path.basename(xml_path)

            with gzip.open(papers_jsonl_gz, "wt", encoding="utf-8") as w:
                context = etree.iterparse(
                    xml_rel,
                    events=("end",),
                    tag=tuple(PUB_TAGS),
                    load_dtd=True,
                    resolve_entities=True,  # <-- critical
                    no_network=True,
                    huge_tree=True,
                )

                for _, elem in context:
                    y = _year_of(elem)
                    if y < args.year_min or y > args.year_max:
                        elem.clear()
                        continue

                    v = _venue_of(elem)
                    if not v:
                        elem.clear()
                        continue

                    authors = _authors_of(elem, args.max_authors_per_paper)
                    if not authors:
                        elem.clear()
                        continue

                    title = elem.findtext("title") or ""
                    terms = _tokenize_title(title, args.max_terms_per_paper)

                    rec = {"authors": authors, "venue": v, "terms": terms, "year": y}
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    for a in set(authors):
                        author_freq[a] += 1
                    venue_freq[v] += 1
                    for t in set(terms):
                        term_df[t] += 1

                    selected += 1
                    if selected >= args.max_papers:
                        break

                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
        finally:
            os.chdir(orig_cwd)

        print(f"[PARSE1] selected papers: {selected}, wrote {papers_jsonl_gz}")
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump({
                "selected_papers": selected,
                "unique_authors": len(author_freq),
                "unique_venues": len(venue_freq),
                "unique_terms": len(term_df),
            }, f, indent=2)
        print(f"[PARSE1] stats -> {stats_json}")
    else:
        print(f"[COUNT] recompute counters from {papers_jsonl_gz} ...")
        with gzip.open(papers_jsonl_gz, "rt", encoding="utf-8") as r:
            for line in r:
                rec = json.loads(line)
                authors = rec.get("authors", [])
                v = rec.get("venue", "")
                terms = rec.get("terms", [])
                for a in set(authors):
                    author_freq[a] += 1
                if v:
                    venue_freq[v] += 1
                for t in set(terms):
                    term_df[t] += 1
                selected += 1
        print(f"[COUNT] papers={selected}, authors={len(author_freq)}, venues={len(venue_freq)}, terms={len(term_df)}")

    # ---- vocab selection ----
    allowed_authors = [a for a, f in author_freq.items() if f >= args.author_min_df]
    allowed_authors.sort(key=lambda a: (-author_freq[a], a))
    allowed_authors = allowed_authors[:args.max_authors]
    author2id = {a: i for i, a in enumerate(allowed_authors)}

    allowed_venues = [v for v, f in venue_freq.items() if f >= args.venue_min_df]
    allowed_venues.sort(key=lambda v: (-venue_freq[v], v))
    allowed_venues = allowed_venues[:args.max_venues]
    venue2id = {v: i for i, v in enumerate(allowed_venues)}

    allowed_terms = [t for t, df in term_df.items() if args.term_min_df <= df <= args.term_max_df]
    allowed_terms.sort(key=lambda t: (-term_df[t], t))
    allowed_terms = allowed_terms[:args.max_terms]
    term2id = {t: i for i, t in enumerate(allowed_terms)}

    print(f"[VOCAB] authors={len(author2id)} venues={len(venue2id)} terms={len(term2id)}")

    # ---- pass2: enforce caps and build COO edges ----
    paper_id = 0
    author_used = collections.Counter()  # aid -> papers contributed
    venue_used  = collections.Counter()  # vid -> papers contributed

    PA_rows: List[int] = []
    PA_cols: List[int] = []
    PT_rows: List[int] = []
    PT_cols: List[int] = []
    PC_rows: List[int] = []
    PC_cols: List[int] = []

    # For APCPA upper-bound estimate: track unique author-per-venue
    venue_author_sets: Dict[int, set] = collections.defaultdict(set) if args.estimate_apcpa else None

    with gzip.open(papers_jsonl_gz, "rt", encoding="utf-8") as r:
        for line in r:
            rec = json.loads(line)

            v = rec.get("venue", "")
            if v not in venue2id:
                continue
            vid = venue2id[v]
            if venue_used[vid] >= args.venue_max_papers:
                continue

            kept_aids = []
            for a in rec.get("authors", []):
                if a not in author2id:
                    continue
                aid = author2id[a]
                if author_used[aid] >= args.author_max_papers:
                    continue
                kept_aids.append(aid)

            kept_aids = list(dict.fromkeys(kept_aids))
            if not kept_aids:
                continue

            kept_tids = []
            for t in rec.get("terms", []):
                tid = term2id.get(t, None)
                if tid is not None:
                    kept_tids.append(tid)
            kept_tids = list(dict.fromkeys(kept_tids))

            pid = paper_id
            paper_id += 1

            # PC
            PC_rows.append(pid); PC_cols.append(vid)
            venue_used[vid] += 1

            # PA
            for aid in kept_aids:
                PA_rows.append(pid); PA_cols.append(aid)
                author_used[aid] += 1
                if venue_author_sets is not None:
                    venue_author_sets[vid].add(aid)

            # PT
            for tid in kept_tids:
                PT_rows.append(pid); PT_cols.append(tid)

            if paper_id >= args.max_papers:
                break

    P = paper_id
    A = len(author2id)
    T = len(term2id)
    C = len(venue2id)

    print(f"[EDGES] P={P} A={A} T={T} C={C}")
    print(f"[NNZ] nnz(PA)={len(PA_rows)} nnz(PT)={len(PT_rows)} nnz(PC)={len(PC_rows)}")

    # ---- optional compact IDs ----
    if args.compact_ids:
        usedA = sorted(set(PA_cols))
        usedT = sorted(set(PT_cols))
        usedC = sorted(set(PC_cols))

        mapA = {old: new for new, old in enumerate(usedA)}
        mapT = {old: new for new, old in enumerate(usedT)}
        mapC = {old: new for new, old in enumerate(usedC)}

        PA_cols = [mapA[x] for x in PA_cols]
        PT_cols = [mapT[x] for x in PT_cols]
        PC_cols = [mapC[x] for x in PC_cols]

        if venue_author_sets is not None:
            new_sets = collections.defaultdict(set)
            for old_vid, s in venue_author_sets.items():
                if old_vid in mapC:
                    new_vid = mapC[old_vid]
                    new_sets[new_vid] = {mapA[aid] for aid in s if aid in mapA}
            venue_author_sets = new_sets

        A = len(usedA)
        T = len(usedT)
        C = len(usedC)
        print(f"[COMPACT] A={A} T={T} C={C}")

    # ---- estimate APCPA upper-bound ----
    if venue_author_sets is not None:
        sizes = [len(s) for s in venue_author_sets.values()]
        sizes.sort(reverse=True)
        top5 = sizes[:5]
        ub = sum(x*x for x in sizes)
        print(f"[EST] APCPA nnz upper-bound ~ sum_v |A_v|^2 = {ub}")
        print(f"[EST] top venue |A_v| (largest 5): {top5}")
        if ub > args.max_apcpa_est:
            print(f"[WARN] APCPA upper-bound {ub} > max_apcpa_est {args.max_apcpa_est}. "
                  f"Consider lowering --venue_max_papers or --author_max_papers.")

    # ---- build CSR ----
    PA_row, PA_col, PA_val = coo_to_csr_numpy(P, A, PA_rows, PA_cols)
    PT_row, PT_col, PT_val = coo_to_csr_numpy(P, T, PT_rows, PT_cols)
    PC_row, PC_col, PC_val = coo_to_csr_numpy(P, C, PC_rows, PC_cols)

    AP_row, AP_col, AP_val = coo_to_csr_numpy(A, P, PA_cols, PA_rows)  # A x P
    TP_row, TP_col, TP_val = coo_to_csr_numpy(T, P, PT_cols, PT_rows)  # T x P
    CP_row, CP_col, CP_val = coo_to_csr_numpy(C, P, PC_cols, PC_rows)  # C x P

    import numpy as np
    def write_csr(prefix: str, row: List[int], col: List[int], val: List[float]):
        np.asarray(row, dtype=np.int32).tofile(os.path.join(out_dir, f"{prefix}_row_offsets.i32"))
        np.asarray(col, dtype=np.int32).tofile(os.path.join(out_dir, f"{prefix}_col_indices.i32"))
        np.asarray(val, dtype=np.float32).tofile(os.path.join(out_dir, f"{prefix}_values.f32"))

    write_csr("PA", PA_row, PA_col, PA_val)
    write_csr("PT", PT_row, PT_col, PT_val)
    write_csr("PC", PC_row, PC_col, PC_val)

    write_csr("AP", AP_row, AP_col, AP_val)
    write_csr("TP", TP_row, TP_col, TP_val)
    write_csr("CP", CP_row, CP_col, CP_val)

    meta = {
        "P": P, "A": A, "T": T, "C": C,
        "PA": {"shape": [P, A], "nnz": len(PA_col),
               "row_offsets": "PA_row_offsets.i32", "col_indices": "PA_col_indices.i32", "values": "PA_values.f32"},
        "PT": {"shape": [P, T], "nnz": len(PT_col),
               "row_offsets": "PT_row_offsets.i32", "col_indices": "PT_col_indices.i32", "values": "PT_values.f32"},
        "PC": {"shape": [P, C], "nnz": len(PC_col),
               "row_offsets": "PC_row_offsets.i32", "col_indices": "PC_col_indices.i32", "values": "PC_values.f32"},
        "AP": {"shape": [A, P], "nnz": len(AP_col),
               "row_offsets": "AP_row_offsets.i32", "col_indices": "AP_col_indices.i32", "values": "AP_values.f32"},
        "TP": {"shape": [T, P], "nnz": len(TP_col),
               "row_offsets": "TP_row_offsets.i32", "col_indices": "TP_col_indices.i32", "values": "TP_values.f32"},
        "CP": {"shape": [C, P], "nnz": len(CP_col),
               "row_offsets": "CP_row_offsets.i32", "col_indices": "CP_col_indices.i32", "values": "CP_values.f32"},
        "caps": {
            "year_min": args.year_min, "year_max": args.year_max,
            "max_papers": args.max_papers,
            "max_authors": args.max_authors,
            "max_venues": args.max_venues,
            "max_terms": args.max_terms,
            "max_authors_per_paper": args.max_authors_per_paper,
            "max_terms_per_paper": args.max_terms_per_paper,
            "author_min_df": args.author_min_df,
            "venue_min_df": args.venue_min_df,
            "author_max_papers": args.author_max_papers,
            "venue_max_papers": args.venue_max_papers,
            "term_min_df": args.term_min_df,
            "term_max_df": args.term_max_df,
            "compact_ids": bool(args.compact_ids),
            "estimate_apcpa": bool(args.estimate_apcpa),
            "max_apcpa_est": args.max_apcpa_est,
        }
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] wrote CSR binaries (PA/PT/PC + AP/TP/CP) + meta.json in {out_dir}")

if __name__ == "__main__":
    main()
