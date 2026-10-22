from __future__ import annotations

import collections
import csv
import html
import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import AUDIT_PROMPT_VERSION, DECISION_POLICY_VERSION, SCHEMA_VERSION
from .common import PROJECT_ROOT, atomic_write_json, atomic_write_jsonl, atomic_write_text, iter_jsonl, read_record, utc_now


def _final_status(
    program, vision, vlm, task_id: str = "", *, vision_required: bool = True,
    vlm_all_pairs: bool = False,
) -> tuple[str, list[str], list[str]]:
    if not program:
        return "pending", [], ["programmatic_missing"]
    reject = list(program.get("reject_reasons", []))
    program_candidate_reasons = {
        "programmatic_foreground_layer_effect_not_over_subject",
        "programmatic_target_effect_too_weak",
    }
    review: list[str] = [
        reason for reason in program.get("review_reasons", [])
        if reason in program_candidate_reasons
    ]
    if reject:
        return "reject", sorted(set(reject)), sorted(set(review))
    if vision_required and not vision:
        return "pending", [], sorted(set(review + ["vision_missing"]))
    if vision_required and vision:
        reject.extend(vision.get("hard_reject_reasons", []))
    if reject:
        return "reject", sorted(set(reject)), sorted(set(review))
    # DINO/SAM/bbox/edge and effect-strength flags are diagnostics, not final
    # gates. VLM is used for generated pairs to reconcile subject change,
    # occlusion, background change, and exact-prompt compliance.
    needs_vlm = vlm_all_pairs or str(program.get("method", "")) != "hybrid"
    if needs_vlm and not vlm:
        return "pending", [], sorted(set(review + ["vlm_required_but_missing"]))
    if vlm:
        parsed_vlm = vlm.get("vlm") or {}
        if (
            vlm.get("schema_version") == SCHEMA_VERSION
            and _nested({"vlm": parsed_vlm}, "vlm", "degradation_alignment", "matches_full_prompt") is True
            and _nested({"vlm": parsed_vlm}, "vlm", "degradation_alignment", "affected_scope_respected") is True
            and _nested({"vlm": parsed_vlm}, "vlm", "subject_visibility", "borderline_visibility") is False
            and _nested({"vlm": parsed_vlm}, "vlm", "gt_quality", "preexisting_occlusion") == "none"
            and float(_nested({"vlm": parsed_vlm}, "vlm", "degradation_alignment", "confidence", default=0.0)) >= 0.9
        ):
            # The mask-derived foreground ratio is a noisy routing proxy. It
            # can only be cleared by explicit, high-confidence, non-borderline
            # three-image evidence; ambiguous cases retain the review signal.
            review = [
                reason for reason in review
                if reason != "programmatic_foreground_layer_effect_not_over_subject"
            ]
        reject.extend(vlm.get("reject_reasons", []))
        review.extend(vlm.get("review_reasons", []))
    if reject:
        return "reject", sorted(set(reject)), sorted(set(review))
    return ("review" if review else "pass"), [], sorted(set(review))


def finalize(args, thresholds) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overrides = _load_adjudication_overrides(args.adjudication_overrides)
    records = []
    status_counts: collections.Counter[str] = collections.Counter()
    task_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    fine_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    reasons: collections.Counter[str] = collections.Counter()
    for row in iter_jsonl(args.manifest):
        pair_id = row["pair_id"]
        program = read_record(args.output_root, "programmatic", pair_id)
        vision = (
            read_record(args.output_root, "vision", pair_id)
            if not bool(getattr(args, "skip_vision", False)) else None
        )
        vlm = read_record(args.output_root, "vlm", pair_id)
        status, reject, review = _final_status(
            program, vision, vlm, str(row.get("task_id", "")),
            vision_required=not bool(getattr(args, "skip_vision", False)),
            vlm_all_pairs=bool(getattr(args, "vlm_all_pairs", False)),
        )
        adjudication = overrides.get(pair_id)
        if adjudication:
            status = str(adjudication.get("quality_status", status))
            reject = list(adjudication.get("reject_reasons", []))
            review = list(adjudication.get("review_reasons", review))
        decision = "candidate" if status == "review" else status
        all_reasons = set(reject + review)
        automatic_moderate_occlusion = "vlm_subject_moderately_occluded" in all_reasons and not adjudication
        automatic_severe_occlusion = "vlm_subject_severely_occluded" in all_reasons and not adjudication
        automatic_prompt_mismatch = bool({
            "vlm_prompt_degradation_mismatch",
            "vlm_affected_scope_mismatch", "vlm_effect_not_applied_to_subject",
            "vlm_target_degradation_missing", "vlm_target_effect_too_weak",
            "programmatic_foreground_layer_effect_not_over_subject",
        } & all_reasons) and not adjudication
        audit_task_id = "occlusion_heavy" if automatic_severe_occlusion else (
            "occlusion_medium" if automatic_moderate_occlusion else row.get("task_id")
        )
        audit_degradation_type = "occlusion" if (automatic_severe_occlusion or automatic_moderate_occlusion) else row.get("degradation_type")
        selection_bucket = (
            "prompt_mismatch_candidate" if automatic_prompt_mismatch and status != "reject"
            else "occlusion_candidate" if automatic_moderate_occlusion and status != "reject"
            else "occlusion_reject" if automatic_severe_occlusion
            else decision
        )
        candidate_buckets = []
        if automatic_prompt_mismatch:
            candidate_buckets.append("prompt_mismatch_candidate")
        if "vlm_subject_visibility_borderline" in all_reasons:
            candidate_buckets.append("visibility_candidate")
        if {"vlm_low_confidence", "vlm_evidence_insufficient", "vlm_evidence_conflict"} & all_reasons:
            candidate_buckets.append("uncertain_evidence_candidate")
        if automatic_moderate_occlusion:
            candidate_buckets.append("occlusion_candidate")
        if automatic_severe_occlusion:
            candidate_buckets.append("occlusion_reject")
        for reason in reject + review:
            reasons[reason] += 1
        record = {
            **row,
            "quality_schema_version": SCHEMA_VERSION,
            "quality_status": status,
            "decision": decision,
            "reason_codes": sorted(set(reject + review)),
            "confidence": _nested(vlm or {}, "vlm", "overall_confidence", default=1.0 if status == "pass" and not vlm else 0.0),
            "reject_reasons": reject,
            "review_reasons": review,
            "programmatic_audit": program,
            "vision_audit": vision,
            "vlm_audit": vlm,
            "threshold_version": thresholds["schema_version"],
            "decision_policy_version": DECISION_POLICY_VERSION,
            "audit_prompt_version": AUDIT_PROMPT_VERSION,
            "programmatic_mode": str(getattr(args, "programmatic_mode", "full")),
            "vision_enabled": not bool(getattr(args, "skip_vision", False)),
            "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
            "model": (vlm or {}).get("model"),
            "temperature": (vlm or {}).get("temperature"),
            "input_image_paths": (vlm or {}).get("input_image_paths") or (program or {}).get("paths", {}),
            "raw_vlm_response": (vlm or {}).get("raw_vlm_response", ""),
            "normalized_vlm_result": (vlm or {}).get("vlm"),
            "decision_source": "manual_adjudication" if adjudication else (
                "hard_rule" if reject and program and program.get("reject_reasons") else
                (vlm or {}).get("decision_source", "combined_policy")
            ),
            "audit_task_id": str(adjudication.get("audit_task_id", audit_task_id)) if adjudication else audit_task_id,
            "audit_degradation_type": str(adjudication.get("audit_degradation_type", audit_degradation_type)) if adjudication else audit_degradation_type,
            "selection_bucket": str(adjudication.get("selection_bucket", selection_bucket)) if adjudication else selection_bucket,
            "candidate_buckets": list(adjudication.get("candidate_buckets", candidate_buckets)) if adjudication else candidate_buckets,
            "manual_adjudication": adjudication,
        }
        records.append(record)
        status_counts[status] += 1
        task_counts[str(record["audit_task_id"])][status] += 1
        fine_counts[f"{row['primary_coarse']}/{row['primary_fine']}"][status] += 1
    manifests = args.output_root / "manifests"
    atomic_write_jsonl(manifests / "pair_quality_audit.jsonl", records)
    for status in ("pass", "review", "reject", "pending"):
        atomic_write_jsonl(manifests / f"pair_quality_{status}.jsonl", (row for row in records if row["quality_status"] == status))
    atomic_write_jsonl(
        manifests / "pair_quality_candidate.jsonl",
        (row for row in records if row["decision"] == "candidate"),
    )
    atomic_write_jsonl(
        manifests / "pair_quality_occlusion_candidates.jsonl",
        (row for row in records if "occlusion_candidate" in row.get("candidate_buckets", [])),
    )
    atomic_write_jsonl(
        manifests / "pair_quality_occlusion_rejects.jsonl",
        (row for row in records if "occlusion_reject" in row.get("candidate_buckets", [])),
    )
    atomic_write_jsonl(
        manifests / "pair_quality_prompt_mismatch_candidates.jsonl",
        (row for row in records if "prompt_mismatch_candidate" in row.get("candidate_buckets", [])),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "threshold_version": thresholds["schema_version"],
        "decision_policy_version": DECISION_POLICY_VERSION,
        "audit_prompt_version": AUDIT_PROMPT_VERSION,
        "programmatic_mode": str(getattr(args, "programmatic_mode", "full")),
        "vision_enabled": not bool(getattr(args, "skip_vision", False)),
        "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
        "generated_at": utc_now(),
        "manifest": str(args.manifest.resolve()),
        "total": len(records),
        "status": dict(status_counts),
        "decision": dict(collections.Counter(str(row["decision"]) for row in records)),
        "selection_bucket": dict(collections.Counter(str(row["selection_bucket"]) for row in records)),
        "candidate_bucket": dict(collections.Counter(
            bucket for row in records for bucket in row.get("candidate_buckets", [])
        )),
        "task_status": {key: dict(value) for key, value in sorted(task_counts.items())},
        "fine_status": {key: dict(value) for key, value in sorted(fine_counts.items())},
        "reason_counts": dict(reasons.most_common()),
    }
    atomic_write_json(args.output_root / "reports/pair_quality_summary.json", summary)
    _write_csv(args.output_root / "reports/pair_quality_statistics.csv", records)
    _write_reason_csv(args.output_root / "reports/reject_review_reason_statistics.csv", reasons)
    _write_html(args, summary, records)
    return summary, records


def _load_adjudication_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("overrides", []) if isinstance(value, dict) else []
    return {str(row["pair_id"]): row for row in rows if isinstance(row, dict) and row.get("pair_id")}


def _nested(record, *keys, default=None):
    value = record
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _write_csv(path: Path, records):
    columns = [
        "pair_id", "sha256", "primary_coarse", "primary_fine", "task_id", "audit_task_id",
        "audit_degradation_type", "selection_bucket", "candidate_buckets", "quality_status",
        "source_gt_dino", "gt_degraded_dino", "source_gt_mask_iou", "gt_degraded_mask_iou",
        "source_gt_center_shift", "gt_degraded_center_shift", "mean_abs_diff", "luma_delta",
        "laplacian_ratio", "foreground_subject_background_diff_ratio", "vlm_confidence",
        "vlm_occluded_fraction", "vlm_prompt_matches", "vlm_affected_scope_respected",
        "vlm_gt_complete_subject_visible", "vlm_degraded_complete_subject_visible",
        "reject_reasons", "review_reasons",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in records:
        writer.writerow({
            "pair_id": row["pair_id"], "sha256": row["sha256"],
            "primary_coarse": row["primary_coarse"], "primary_fine": row["primary_fine"],
            "task_id": row["task_id"], "audit_task_id": row["audit_task_id"],
            "audit_degradation_type": row["audit_degradation_type"],
            "selection_bucket": row["selection_bucket"],
            "candidate_buckets": "|".join(row.get("candidate_buckets", [])),
            "quality_status": row["quality_status"],
            "source_gt_dino": _nested(row, "vision_audit", "dino", "source_gt_similarity"),
            "gt_degraded_dino": _nested(row, "vision_audit", "dino", "gt_degraded_similarity"),
            "source_gt_mask_iou": _nested(row, "vision_audit", "source_to_gt", "mask_iou"),
            "gt_degraded_mask_iou": _nested(row, "vision_audit", "gt_to_degraded", "mask_iou"),
            "source_gt_center_shift": _nested(row, "vision_audit", "source_to_gt", "bbox_center_shift"),
            "gt_degraded_center_shift": _nested(row, "vision_audit", "gt_to_degraded", "bbox_center_shift"),
            "mean_abs_diff": _nested(row, "programmatic_audit", "degradation_metrics", "mean_abs_diff"),
            "luma_delta": _nested(row, "programmatic_audit", "degradation_metrics", "luma_delta"),
            "laplacian_ratio": _nested(row, "programmatic_audit", "degradation_metrics", "laplacian_variance_ratio"),
            "foreground_subject_background_diff_ratio": _nested(row, "programmatic_audit", "degradation_metrics", "foreground_subject_background_diff_ratio"),
            "vlm_confidence": _nested(row, "vlm_audit", "vlm", "overall_confidence"),
            "vlm_occluded_fraction": _nested(row, "vlm_audit", "vlm", "subject_visibility", "estimated_occluded_fraction"),
            "vlm_prompt_matches": _nested(row, "vlm_audit", "vlm", "degradation_alignment", "matches_full_prompt"),
            "vlm_affected_scope_respected": _nested(row, "vlm_audit", "vlm", "degradation_alignment", "affected_scope_respected"),
            "vlm_gt_complete_subject_visible": _nested(row, "vlm_audit", "vlm", "gt_quality", "subject_fully_visible"),
            "vlm_degraded_complete_subject_visible": _nested(row, "vlm_audit", "vlm", "subject_visibility", "usable_for_subject_learning"),
            "reject_reasons": "|".join(row["reject_reasons"]), "review_reasons": "|".join(row["review_reasons"]),
        })
    atomic_write_text(path, buffer.getvalue())


def _write_reason_csv(path, reasons):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["reason", "count"])
    writer.writerows(reasons.most_common())
    atomic_write_text(path, buffer.getvalue())


def _table(mapping):
    rows = []
    for key, counts in mapping.items():
        rows.append(f"<tr><td>{html.escape(key)}</td><td>{counts.get('pass',0)}</td><td>{counts.get('review',0)}</td><td>{counts.get('reject',0)}</td><td>{counts.get('pending',0)}</td></tr>")
    return "".join(rows)


def _url(path: str, report: Path) -> str:
    return quote(os.path.relpath(path, report.parent).replace(os.sep, "/"), safe="/._-")


def _display_path(args, row, paths, name: str) -> Path:
    candidate = Path(str(paths.get(name, "")))
    if candidate.is_file() and str(candidate).startswith(str(PROJECT_ROOT)):
        return candidate
    preview = args.pair_root / "reports/full_50_case_assets" / f"{row['pair_id']}_{name}.webp"
    if preview.is_file():
        return preview
    if name == "source":
        relative = row.get("source_relative_path")
        root = args.dataset_root
    else:
        relative = row.get(f"{name}_relative_path")
        root = args.pair_root
    return root / str(relative or "missing")


def _write_html(args, summary, records):
    report = args.output_root / "reports/pair_quality_report.html"
    representatives = []
    display_records = sorted(records, key=lambda value: ({"reject": 0, "candidate": 1, "pass": 2, "pending": 3}[value["decision"]], value["primary_coarse"], value["primary_fine"], value["pair_id"]))[:200]
    for row in display_records:
        if row["quality_status"] == "pending":
            continue
        program = row.get("programmatic_audit") or {}
        paths = program.get("paths") or {}
        display_paths = {name: _display_path(args, row, paths, name) for name in ("source", "gt", "degraded")}
        if not all(path.is_file() for path in display_paths.values()):
            continue
        images = "".join(f'<figure><img loading="lazy" src="{_url(str(display_paths[name]), report)}"><figcaption>{label}</figcaption></figure>' for name, label in (("source","source"),("gt","GT"),("degraded","degraded")))
        reasons = row["reject_reasons"] or row["review_reasons"]
        source_gt_dino = _nested(row, "vision_audit", "dino", "source_gt_similarity", default="—")
        gt_deg_dino = _nested(row, "vision_audit", "dino", "gt_degraded_similarity", default="—")
        vlm_confidence = _nested(row, "vlm_audit", "vlm", "overall_confidence", default="—")
        occluded_fraction = _nested(row, "vlm_audit", "vlm", "subject_visibility", "estimated_occluded_fraction", default="—")
        foreground_ratio = _nested(row, "programmatic_audit", "degradation_metrics", "foreground_subject_background_diff_ratio", default="—")
        prompt_match = _nested(row, "vlm_audit", "vlm", "degradation_alignment", "matches_full_prompt", default="—")
        complete_visible = _nested(row, "vlm_audit", "vlm", "subject_visibility", "usable_for_subject_learning", default="—")
        evidence, issues = [], []
        for block in ("source_gt_identity", "gt_quality", "degradation_alignment", "subject_visibility"):
            evidence.extend(_nested(row, "vlm_audit", "vlm", block, "evidence", default=[]) or [])
            issues.extend(_nested(row, "vlm_audit", "vlm", block, "issues", default=[]) or [])
        evidence_html = ""
        if evidence or issues:
            evidence_html = (
                '<details><summary>VLM 证据与问题</summary><p><b>证据：</b>'
                + html.escape("；".join(str(x) for x in evidence[:8]) or "—")
                + '</p><p><b>问题：</b>'
                + html.escape("；".join(str(x) for x in issues[:8]) or "—")
                + '</p></details>'
            )
        title = f"{row['primary_coarse']}/{row['primary_fine']} · {row['audit_task_id']}"
        original = ""
        if row.get("manual_adjudication"):
            original = f'<p class="manual">人工复核：{html.escape(str(row["manual_adjudication"].get("note", "已应用人工裁决")))}<br>原生成任务：{html.escape(str(row["task_id"]))}；待选桶：{html.escape(str(row["selection_bucket"]))}</p>'
        vlm_label = "policy bypass" if _nested(row, "vlm_audit", "policy_bypass", default=False) else vlm_confidence
        representatives.append(f'<article data-status="{row["decision"]}"><header><b>{html.escape(title)}</b><span class="{row["decision"]}">{row["decision"]}</span></header><div class="imgs">{images}</div><p class="metrics">DINO source→GT={source_gt_dino} · GT→deg={gt_deg_dino} · VLM={vlm_label} · 遮挡诊断={occluded_fraction} · 主体整体可见={complete_visible} · degradation prompt匹配={prompt_match} · 前景/背景改变量={foreground_ratio}</p>{original}<p>{html.escape("; ".join(reasons) or "all gates passed")}</p>{evidence_html}<code>{row["pair_id"]}</code></article>')
    cards = "".join(f'<div><b>{summary["decision"].get(key,0):,}</b><span>{key}</span></div>' for key in ("pass","candidate","reject","pending"))
    reason_rows = "".join(f"<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>" for reason, count in list(summary["reason_counts"].items())[:40]) or "<tr><td>none</td><td>0</td></tr>"
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TexVerse Pair Quality Audit</title><style>
body{{margin:0;color:#17233d;background:#fff;font-family:Inter,"PingFang SC",sans-serif}}main{{max-width:1500px;margin:auto;padding:34px 28px}}h1{{font-size:34px}}.muted{{color:#667085}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.cards div{{padding:18px;background:#f3f7ff;border-radius:10px}}.cards b,.cards span{{display:block}}.cards b{{font-size:26px;color:#3569df}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #dce3ec;text-align:left}}th{{background:#edf3ff}}.cases{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}article{{border:1px solid #dce3ec;border-radius:12px;padding:12px}}article header{{display:flex;justify-content:space-between}}header span{{padding:3px 8px;border-radius:12px}}.pass{{background:#e7f7ef;color:#087443}}.candidate,.review{{background:#fff5dc;color:#936600}}.reject{{background:#feecec;color:#b42318}}.manual{{padding:8px;border-radius:7px;background:#eef4ff;color:#315bb5}}.imgs{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:9px}}figure{{margin:0}}img{{width:100%;aspect-ratio:1;object-fit:contain;background:#f5f7fa}}figcaption{{text-align:center;font-size:11px;color:#667085}}article p,code,details{{font-size:11px;overflow-wrap:anywhere}}details{{padding:7px;background:#f7f9fc;border-radius:7px;margin:8px 0}}@media(max-width:800px){{.cards,.cases{{grid-template-columns:1fr 1fr}}}}@media(max-width:520px){{.cards,.cases{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>TexVerse GT–退化图像质量审计</h1><p class="muted">基础完整性硬门槛；DINOv2/SAM/几何指标用于诊断；VLM 检查 source→GT 的主体保持，并按 degradation prompt 裁决主体、背景、作用层和退化匹配；不检查 GT prompt 匹配。阈值版本 {html.escape(summary["threshold_version"])}。</p><p class="manual"><b>当前裁决策略：</b>任何单一或组合 crop 任务直接 Reject；反射、光照、阴影、污渍等 prompt 要求的效果允许覆盖主体，只要主体仍能作为完整对象被看清；主体整体不可见或透明效果变成实体遮挡时 Reject；未按完整 degradation prompt 实现的 pair 进入 prompt-mismatch 候选。</p><div class="cards">{cards}</div><h2>退化任务状态</h2><table><thead><tr><th>任务</th><th>Pass</th><th>Review</th><th>Reject</th><th>Pending</th></tr></thead><tbody>{_table(summary["task_status"])}</tbody></table><h2>主要失败/复核原因</h2><table><thead><tr><th>原因</th><th>数量</th></tr></thead><tbody>{reason_rows}</tbody></table><h2>逐 Pair 审计案例（最多 200 条）</h2><div class="cases">{"".join(representatives)}</div></main></body></html>'''
    atomic_write_text(report, page)
