"""Blind pair assignments, auditable human preference counts and Table 4 VLM aggregation."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from .io import iter_jsonl, write_jsonl

CRITERIA = ("identity", "purification", "quality")


def assignments(rows, seed):
    output = []
    seen = set()
    for row in rows:
        if not row.get("case_id") or "ours" not in row["methods"] or len(row["methods"]) < 2:
            raise ValueError("Require a case ID, ours and at least one baseline")
        for baseline in sorted(set(row["methods"]) - {"ours"}):
            key = f"{row['case_id']}:{baseline}"
            if key in seen:
                raise ValueError("Duplicate preference case/baseline")
            seen.add(key)
            order = ["ours", baseline]
            random.Random(f"{seed}:{key}").shuffle(order)
            output.append({"assignment_id": hashlib.sha256(key.encode()).hexdigest()[:20], "case_id": row["case_id"],
                "baseline": baseline, "slot_map": dict(zip(("A", "B"), order)),
                "display": {slot: row["methods"][method] for slot, method in zip(("A", "B"), order)}})
    random.Random(seed).shuffle(output)
    if not output:
        raise ValueError("Empty preference population")
    return output


def human_summary(assignment_rows, ratings):
    index = {r["assignment_id"]: r for r in assignment_rows}
    if len(index) != len(assignment_rows):
        raise ValueError("Duplicate assignment IDs")
    counts = defaultdict(lambda: {c: {"ours": 0, "baseline": 0, "tie": 0} for c in CRITERIA})
    seen, excluded, valid = set(), [], 0
    for row in ratings:
        identity = (row["participant_id"], row["assignment_id"])
        if identity in seen:
            raise ValueError("Duplicate participant/assignment rating")
        seen.add(identity)
        if not row["participant_id"] or row.get("valid") is not True:
            excluded.append({"participant_id": row["participant_id"], "assignment_id": row["assignment_id"],
                             "reason": row.get("exclusion_reason", "not explicitly valid")})
            continue
        assignment = index[row["assignment_id"]]
        if any(row.get(c) not in {"A", "B", "tie"} for c in CRITERIA):
            raise ValueError("Complete A/B/tie choices required on all three criteria")
        valid += 1
        for c in CRITERIA:
            choice = row[c]
            resolved = "tie" if choice == "tie" else ("ours" if assignment["slot_map"][choice] == "ours" else "baseline")
            counts[assignment["baseline"]][c][resolved] += 1
    for baseline in counts.values():
        for c in baseline.values():
            total = c["ours"] + c["baseline"] + c["tie"]
            c.update(valid_ratings=total, ours_preference_percent=100 * c["ours"] / total if total else None)
    return {"valid_ratings": valid, "excluded": excluded, "baselines": dict(counts),
            "tie_policy": "ties included in denominator; not credited as ours wins"}


def vlm_summary(rows):
    groups = defaultdict(list)
    identities = set()
    for r in rows:
        key = (r["case_id"], r["method"], r["domain"], r.get("generator"), r.get("view"))
        if key in identities:
            raise ValueError("Duplicate VLM score identity")
        identities.add(key)
        if r["domain"] not in {"2d", "3d"}:
            raise ValueError("Unknown VLM domain")
        for criterion in CRITERIA[:2] if r["domain"] == "2d" else CRITERIA:
            value = float(r[criterion])
            if not np.isfinite(value) or not 0 <= value <= 10:
                raise ValueError("VLM scores must be finite values in [0,10]")
        groups[(r["method"], r["case_id"])].append(r)
    method_rows = defaultdict(list)
    for (method, case), items in groups.items():
        two = [r for r in items if r["domain"] == "2d"]
        three = [r for r in items if r["domain"] == "3d"]
        required = {(g, v) for g in ("pixal3d", "trellis2") for v in range(4)}
        if len(two) != 1 or {(r["generator"], r["view"]) for r in three} != required or len(three) != 8:
            raise ValueError("Each method/case requires one 2D score and four views from both generators")
        metrics = {c: .5 * (float(two[0][c]) + np.mean([float(r[c]) for r in three])) for c in CRITERIA[:2]}
        metrics["quality"] = np.mean([float(r["quality"]) for r in three])
        method_rows[method].append((case, metrics))
    populations = [set(case for case, _ in values) for values in method_rows.values()]
    if not populations or any(p != populations[0] for p in populations):
        raise ValueError("All methods must cover the same case population")
    return {method: {"cases": len(values), **{c: float(np.mean([m[c] for _, m in values])) for c in CRITERIA}}
            for method, values in method_rows.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=1234)
    prepare.add_argument("--output-dir", type=Path, required=True)
    human = sub.add_parser("human")
    human.add_argument("--assignments", type=Path, required=True)
    human.add_argument("--ratings", type=Path, required=True)
    human.add_argument("--output", type=Path, required=True)
    vlm = sub.add_parser("vlm")
    vlm.add_argument("--scores", type=Path, required=True)
    vlm.add_argument("--judge-model", required=True)
    vlm.add_argument("--judge-revision", required=True)
    vlm.add_argument("--prompt", type=Path, required=True)
    vlm.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        output = assignments(list(iter_jsonl(args.manifest)), args.seed)
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError("Use a new assignment directory")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "private_assignments.jsonl", output)
        write_jsonl(args.output_dir / "blind_display.jsonl", [{k: r[k] for k in ("assignment_id", "display")} for r in output])
        return 0
    if args.command == "human":
        result = human_summary(list(iter_jsonl(args.assignments)), list(iter_jsonl(args.ratings)))
        result["ratings_sha256"] = hashlib.sha256(args.ratings.read_bytes()).hexdigest()
    else:
        result = {"methods": vlm_summary(list(iter_jsonl(args.scores))), "judge_model": args.judge_model,
                  "judge_revision": args.judge_revision, "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
                  "scores_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
