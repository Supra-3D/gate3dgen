from __future__ import annotations

import collections
import html
import json
import time
from pathlib import Path
from typing import Any, Optional

from .common import atomic_write_json, atomic_write_jsonl, iter_jsonl
from .generation import pair_output_dir, utc_now


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _metadata_output_path(metadata: dict[str, Any], key: str, output_root: Path) -> Path:
    relative_key = key.replace("_output", "_relative_path")
    relative = metadata.get(relative_key)
    if relative:
        return output_root / str(relative)
    original = Path(str(metadata.get(key, "")))
    if "pairs" in original.parts:
        index = original.parts.index("pairs")
        remapped = output_root / Path(*original.parts[index:])
        if remapped.is_file():
            return remapped
    return original


def _relative_to_root(path: Path, output_root: Path) -> Optional[str]:
    try:
        return path.relative_to(output_root).as_posix()
    except ValueError:
        return None


def _success_outputs_present(metadata: dict[str, Any], output_root: Path) -> bool:
    if metadata.get("basic_audit", {}).get("status") != "pass":
        return False
    for key in ("gt_output", "degraded_output"):
        path = _metadata_output_path(metadata, key, output_root)
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    mask_output = metadata.get("mask_output") or metadata.get("mask_relative_path")
    if not mask_output:
        return True
    mask_path = _metadata_output_path(metadata, "mask_output", output_root)
    return mask_path.is_file() and mask_path.stat().st_size > 0


def collect_report(plan_path: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    started = time.monotonic()
    buckets: dict[str, list[dict[str, Any]]] = {"pass": [], "reject": [], "pending": []}
    task_status: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    split_status: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    slot_status: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    category_status: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    asset_status: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    failure_reasons: collections.Counter[str] = collections.Counter()
    gpu_status: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    gpu_elapsed: collections.Counter[str] = collections.Counter()
    total_elapsed = 0.0
    attempt_count = 0
    retry_count = 0
    for row in iter_jsonl(plan_path):
        metadata_path = pair_output_dir(output_root, row) / "metadata.json"
        metadata = _read_metadata(metadata_path)
        if metadata and metadata.get("pair_id") == row["pair_id"] and metadata.get("status") == "success" and _success_outputs_present(metadata, output_root):
            status = "pass"
            gt_path = _metadata_output_path(metadata, "gt_output", output_root)
            degraded_path = _metadata_output_path(metadata, "degraded_output", output_root)
            record = {
                **row,
                "source_path": metadata.get("source_path"),
                "source_image_sha256": metadata.get("source_image_sha256"),
                "gt_path": str(gt_path.absolute()),
                "gt_relative_path": _relative_to_root(gt_path, output_root),
                "degraded_path": str(degraded_path.absolute()),
                "degraded_relative_path": _relative_to_root(degraded_path, output_root),
                "gt_sha256": metadata.get("basic_audit", {}).get("gt_sha256"),
                "degraded_sha256": metadata.get("basic_audit", {}).get("degraded_sha256"),
                "metadata_path": str(metadata_path.resolve()),
                "accepted_attempt_seed": metadata.get("accepted_attempt_seed"),
                "basic_audit_status": metadata.get("basic_audit", {}).get("status"),
            }
            buckets[status].append(record)
        elif metadata and metadata.get("pair_id") == row["pair_id"] and metadata.get("status") == "reject":
            status = "reject"
            attempts = metadata.get("attempts", [])
            last_error = attempts[-1].get("error", "unknown") if attempts else "unknown"
            failure_reasons[last_error.split(":", 1)[0]] += 1
            buckets[status].append({**row, "metadata_path": str(metadata_path.resolve()), "last_error": last_error})
        else:
            status = "pending"
            buckets[status].append(row)
        attempts = (metadata or {}).get("attempts", [])
        attempt_count += len(attempts)
        retry_count += max(0, len(attempts) - 1)
        item_elapsed = sum(float(item.get("elapsed_seconds", 0.0)) for item in attempts)
        total_elapsed += item_elapsed
        if metadata and metadata.get("gpu") is not None:
            gpu = str(metadata["gpu"])
            gpu_status[gpu][status] += 1
            gpu_status[gpu]["attempts"] += len(attempts)
            gpu_elapsed[gpu] += item_elapsed
        task_status[row["task_id"]][status] += 1
        split_status[row["split"]][status] += 1
        slot_status[int(row["view_slot"])][status] += 1
        category_status[f"{row['primary_coarse']}/{row['primary_fine']}"][status] += 1
        asset_status[row["sha256"]][status] += 1
    incomplete_assets = []
    for asset_id, counts in asset_status.items():
        if counts["pass"] != 12:
            incomplete_assets.append({"sha256": asset_id, **dict(counts)})
    completed = len(buckets["pass"]) + len(buckets["reject"])
    observed_gpu_count = len(gpu_status)
    mean_pair_seconds = total_elapsed / completed if completed else None
    eta_seconds = (
        len(buckets["pending"]) * mean_pair_seconds / observed_gpu_count
        if mean_pair_seconds is not None and observed_gpu_count else None
    )
    gpu_throughput = {}
    for gpu, counts in sorted(gpu_status.items()):
        gpu_completed = counts["pass"] + counts["reject"]
        gpu_throughput[gpu] = {
            **dict(counts),
            "attempt_elapsed_seconds": round(gpu_elapsed[gpu], 3),
            "completed_pairs_per_gpu_hour": round(gpu_completed * 3600 / gpu_elapsed[gpu], 3) if gpu_elapsed[gpu] else None,
        }
    summary = {
        "generated_at": utc_now(),
        "plan_path": str(plan_path.resolve()),
        "planned": sum(len(items) for items in buckets.values()),
        "pass": len(buckets["pass"]),
        "reject": len(buckets["reject"]),
        "pending": len(buckets["pending"]),
        "completed": completed,
        "assets": len(asset_status),
        "complete_assets": len(asset_status) - len(incomplete_assets),
        "incomplete_assets": len(incomplete_assets),
        "attempt_count": attempt_count,
        "retry_count": retry_count,
        "attempt_elapsed_seconds": round(total_elapsed, 3),
        "mean_pair_seconds": round(mean_pair_seconds, 3) if mean_pair_seconds is not None else None,
        "observed_gpu_count": observed_gpu_count,
        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "gpu_throughput": gpu_throughput,
        "task_status": {key: dict(value) for key, value in sorted(task_status.items())},
        "split_status": {key: dict(value) for key, value in sorted(split_status.items())},
        "view_slot_status": {str(key): dict(value) for key, value in sorted(slot_status.items())},
        "category_status": {key: dict(value) for key, value in sorted(category_status.items())},
        "failure_reasons": dict(failure_reasons.most_common()),
        "scan_seconds": round(time.monotonic() - started, 3),
    }
    buckets["incomplete_assets"] = incomplete_assets
    return summary, buckets


def _status_table(title: str, mapping: dict[str, dict[str, int]]) -> str:
    rows = []
    for key, counts in mapping.items():
        rows.append(
            f"<tr><td>{html.escape(str(key))}</td><td>{counts.get('pass', 0)}</td>"
            f"<td>{counts.get('reject', 0)}</td><td>{counts.get('pending', 0)}</td></tr>"
        )
    return (
        f"<h2>{html.escape(title)}</h2><table><thead><tr><th>分组</th><th>Pass</th>"
        f"<th>Reject</th><th>Pending</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def write_report(plan_path: Path, output_root: Path) -> dict[str, Any]:
    summary, buckets = collect_report(plan_path, output_root)
    manifests = output_root / "manifests"
    reports = output_root / "reports"
    atomic_write_jsonl(manifests / "all_pass.jsonl", buckets["pass"])
    atomic_write_jsonl(manifests / "rejects.jsonl", buckets["reject"])
    atomic_write_jsonl(manifests / "pending.jsonl", buckets["pending"])
    for split in ("train", "validation", "test"):
        atomic_write_jsonl(manifests / f"{split}.jsonl", (row for row in buckets["pass"] if row["split"] == split))
    pilot_pass = [row for row in buckets["pass"] if row.get("pilot")]
    atomic_write_jsonl(manifests / "pilot_pass.jsonl", pilot_pass)
    atomic_write_jsonl(manifests / "incomplete_assets.jsonl", buckets["incomplete_assets"])
    atomic_write_jsonl(output_root / "audit/audit_input.jsonl", buckets["pass"])
    atomic_write_json(reports / "generation_report.json", summary)
    cards = "".join(
        f"<div><b>{value:,}</b><span>{label}</span></div>" for label, value in (
            ("Planned", summary["planned"]), ("Pass", summary["pass"]),
            ("Reject", summary["reject"]), ("Pending", summary["pending"]),
            ("Complete assets", summary["complete_assets"]),
        )
    )
    failure_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in summary["failure_reasons"].items()
    ) or "<tr><td>None</td><td>0</td></tr>"
    gpu_rows = "".join(
        f"<tr><td>{html.escape(gpu)}</td><td>{counts.get('pass', 0)}</td>"
        f"<td>{counts.get('reject', 0)}</td><td>{counts.get('attempts', 0)}</td>"
        f"<td>{counts.get('completed_pairs_per_gpu_hour') or '—'}</td></tr>"
        for gpu, counts in summary["gpu_throughput"].items()
    ) or "<tr><td colspan=\"5\">尚无已执行 GPU 记录</td></tr>"
    eta = f"{summary['eta_seconds'] / 3600:.2f} 小时" if summary["eta_seconds"] is not None else "待产生执行记录后估算"
    page = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>TexVerse Pair Generation Report</title>
<style>body{{font-family:Inter,'PingFang SC',sans-serif;color:#17233d;margin:0;background:#fff}}main{{max-width:1280px;margin:auto;padding:38px 28px}}h1{{font-size:38px}}h2{{margin-top:32px}}.cards{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}.cards div{{background:#f3f7ff;padding:20px;border-radius:12px}}.cards b,.cards span{{display:block}}.cards b{{font-size:28px;color:#3569df}}.cards span{{color:#667085;margin-top:5px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #dce3ec}}th{{background:#edf3ff}}.note{{color:#667085}}</style></head><body><main>
<h1>TexVerse GT–退化生成审计</h1><p class=\"note\">更新时间：{html.escape(summary['generated_at'])}；47,604 是计划数，训练使用实际 Pass manifest。</p>
<div class=\"cards\">{cards}</div>
<p class=\"note\">Attempts：{summary['attempt_count']:,}；Retries：{summary['retry_count']:,}；按当前已观测吞吐估算剩余时间：{eta}。</p>
{_status_table('退化类型', summary['task_status'])}
{_status_table('四视角槽', summary['view_slot_status'])}
{_status_table('数据划分', summary['split_status'])}
<h2>GPU 吞吐</h2><table><thead><tr><th>GPU</th><th>Pass</th><th>Reject</th><th>Attempts</th><th>完成 pairs/GPU·h</th></tr></thead><tbody>{gpu_rows}</tbody></table>
<h2>失败原因</h2><table><thead><tr><th>错误类型</th><th>数量</th></tr></thead><tbody>{failure_rows}</tbody></table>
</main></body></html>"""
    from .common import atomic_write_text
    atomic_write_text(reports / "generation_report.html", page)
    return summary
