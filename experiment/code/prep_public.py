#!/usr/bin/env python3
"""Spider dev and BIRD Mini-Dev v1 (SQLite) items, plus export to the official evaluators' formats.

Revised from prior-code/prep_public.py: the dataset roots are arguments (PC and Jetson paths
differ), ddl() is unchanged, and the corrected Mini-Dev (Arcwise-Plat-SQL, Jin et al.) gold
file can be written for re-scoring the same predictions.

    python prep_public.py items  --spider-root S --bird-root B --out data/public
    python prep_public.py export --which spider|bird --spider-root S --bird-root B --pred preds.jsonl \
        --ids ids.txt --out pred.txt
    python prep_public.py arcwise --bird-root B --arcwise arcwise_plat_sql_only_with_diff.json --out dir
"""
import argparse
import json
import sqlite3
from pathlib import Path


def ddl(db_path: Path) -> str:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    stmts = [r[0].strip() for r in conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
    conn.close()
    return ";\n\n".join(stmts) + ";"


def spider_items(root: Path):
    for i, ex in enumerate(json.loads((root / "dev.json").read_text(encoding="utf-8"))):
        db = ex["db_id"]
        yield {"id": f"spider_dev_{i:04d}", "question": ex["question"], "db_id": db,
               "sql": ex["query"], "db_path": str(root / "database" / db / f"{db}.sqlite")}


def bird_items(root: Path):
    for i, ex in enumerate(json.loads((root / "mini_dev_sqlite.json").read_text(encoding="utf-8"))):
        db = ex["db_id"]
        yield {"id": f"bird_minidev_{i:03d}", "question": ex["question"], "evidence": ex["evidence"],
               "db_id": db, "sql": ex["SQL"], "difficulty": ex["difficulty"],
               "question_id": ex["question_id"],
               "db_path": str(root / "dev_databases" / db / f"{db}.sqlite")}


def items(args):
    args.out.mkdir(parents=True, exist_ok=True)
    for name, its in (("spider_dev", spider_items(args.spider_root)), ("bird_minidev", bird_items(args.bird_root))):
        cache, n = {}, 0
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for ex in its:
                if ex["db_path"] not in cache:
                    cache[ex["db_path"]] = ddl(Path(ex["db_path"]))
                ex["schema"] = cache[ex["db_path"]]
                ex.pop("db_path")
                f.write(json.dumps(ex) + "\n")
                n += 1
        print(f"{name}: {n} questions over {len(cache)} databases")


def load_preds(path):
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                preds[p["id"]] = " ".join((p.get("sql") or "").split()) or "SELECT 1"
    return preds


def export(args):
    """Write predictions in evaluator order. --ids restricts to a subset (pilot), in dataset order."""
    preds = load_preds(args.pred)
    keep = set(args.ids.read_text().split()) if args.ids else None
    if args.which == "spider":
        its = [ex for ex in spider_items(args.spider_root) if keep is None or ex["id"] in keep]
        args.out.write_text("\n".join(preds.get(ex["id"], "SELECT 1") for ex in its) + "\n", encoding="utf-8")
        if args.gold_out:  # matching gold subset: "SQL\tdb_id" per line, as dev_gold.sql
            args.gold_out.write_text("\n".join(f"{ex['sql']}\t{ex['db_id']}" for ex in its) + "\n", encoding="utf-8")
    else:
        its = [ex for ex in bird_items(args.bird_root) if keep is None or ex["id"] in keep]
        out = {str(k): f"{preds.get(ex['id'], 'SELECT 1')}\t----- bird -----\t{ex['db_id']}"
               for k, ex in enumerate(its)}
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        if args.gold_out:
            gold = json.loads(args.gold_src.read_text(encoding="utf-8")) if args.gold_src else None
            lines, meta = [], []
            src = {str(g["question_id"]): g for g in gold} if gold else None
            for ex in its:
                sql = src[str(ex["question_id"])]["SQL"] if src else ex["sql"]
                lines.append(f"{' '.join(sql.split())}\t{ex['db_id']}")
                meta.append({"question_id": ex["question_id"], "db_id": ex["db_id"], "difficulty": ex["difficulty"]})
            args.gold_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            args.gold_out.with_suffix(".meta.jsonl").write_text(
                "".join(json.dumps(m) + "\n" for m in meta), encoding="utf-8")
    missing = [ex["id"] for ex in its if ex["id"] not in preds]
    print(f"wrote {args.out} ({len(its)} items); {len(missing)} missing predictions scored as wrong")


def arcwise(args):
    """Check Arcwise-Plat-SQL against v1 (same questions/evidence), list the 498 covered ids."""
    v1 = {str(ex["question_id"]): ex for ex in bird_items(args.bird_root)}
    arc = json.loads(args.arcwise.read_text(encoding="utf-8"))
    diffs, changed, missing = [], 0, []
    for a in arc:
        q = str(a["question_id"])
        if q not in v1:
            missing.append(q)
            continue
        b = v1[q]
        for k_a, k_b in (("question", "question"), ("evidence", "evidence"), ("db_id", "db_id")):
            if a.get(k_a) != b.get(k_b):
                diffs.append({"question_id": q, "field": k_a})
        changed += " ".join(a["SQL"].split()) != " ".join(b["sql"].split())
    args.out.mkdir(parents=True, exist_ok=True)
    rep = {"n_arcwise": len(arc), "n_v1": len(v1), "not_in_v1": missing, "field_diffs": diffs,
           "gold_sql_changed": changed, "v1_ids_not_in_arcwise": sorted(set(v1) - {str(a["question_id"]) for a in arc})}
    (args.out / "arcwise_check.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["items", "export", "arcwise"])
    ap.add_argument("--spider-root", type=Path)
    ap.add_argument("--bird-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--which", choices=["spider", "bird"])
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--ids", type=Path)
    ap.add_argument("--gold-out", type=Path)
    ap.add_argument("--gold-src", type=Path, help="alternative BIRD gold (e.g. Arcwise-Plat-SQL json)")
    ap.add_argument("--arcwise", type=Path)
    args = ap.parse_args()
    {"items": items, "export": export, "arcwise": arcwise}[args.cmd](args)


if __name__ == "__main__":
    main()
