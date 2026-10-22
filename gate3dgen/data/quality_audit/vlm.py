from __future__ import annotations

import base64
import io
import json
import os
import selectors
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from . import AUDIT_PROMPT_VERSION, DECISION_POLICY_VERSION, SCHEMA_VERSION
from .common import atomic_write_json, input_fingerprint, read_record, record_path, stage_rows, utc_now


OCCLUSION_RUBRIC = (
    "半透明遮挡但主体整体可见可以通过。\n"
    "实体遮挡主体关键部分应拒绝。\n"
    "介于半透明反射和实体遮挡之间则进入候选。"
)


def _data_url(path: Path, max_edge: int = 768) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _messages(row, program) -> list[dict[str, Any]]:
    schema = {
        "pair_id": row["pair_id"],
        "source_gt_identity": {
            "subject_consistent": True, "geometry_and_key_parts_preserved": True,
            "viewpoint_compatible": True, "confidence": 0.95, "evidence": [], "issues": [],
        },
        "gt_quality": {
            "subject_fully_visible": True, "subject_severely_out_of_frame": False,
            "preexisting_occlusion": "none", "reflection_severity": "none",
            "rendering_anomaly": "none", "confidence": 0.95, "evidence": [], "issues": [],
        },
        "degradation_alignment": {
            "effect_present": True, "matches_full_prompt": True, "applied_to_subject": True,
            "affected_scope_respected": True, "effect_scope": "foreground",
            "effect_severity": "heavy", "confidence": 0.95, "evidence": [], "issues": [],
        },
        "subject_visibility": {
            "whole_subject_recognizable": True, "overall_silhouette_visible": True,
            "key_parts_mostly_visible": True, "usable_for_subject_learning": True,
            "occluder_type": "transparent", "solid_occluder_present": False,
            "estimated_occluded_fraction": 0.2, "borderline_visibility": False,
            "confidence": 0.95, "evidence": [], "issues": [],
        },
        "evidence_sufficiency": {
            "all_three_images_readable": True, "evidence_conflict": False,
            "notes": "", "confidence": 0.95,
        },
        "unexpected_degradations": [], "overall_confidence": 0.95,
    }
    expected_prompt = str(program.get("expected_degradation_prompt") or row.get("degradation_prompt") or "")
    instruction = f"""You audit a three-image training pair. IMAGE 1 is source albedo, IMAGE 2 is a clean real-scene GT generated from it, IMAGE 3 is the degraded version of IMAGE 2.
Target degradation: {row.get('degradation_type')} ({row.get('task_id')}); affected scope: {program.get('affected_scope')}.
Exact degradation prompt used to create IMAGE 3: {expected_prompt}
The PRIMARY SUBJECT is always the object shared by IMAGE 1 and IMAGE 2, never a newly added object in IMAGE 3.

Return visual observations, not a pass/reject decision. Code will make the final decision from your fields.

Check in this order:
1. SOURCE->GT identity: only identity, geometry, viewpoint, silhouette and key-part preservation. Do NOT check a GT-generation prompt or scene description.
2. GT quality: independently inspect whether GT already has severe reflection, occlusion, rendering failure, or framing that prevents the primary subject being learned as a complete object.
3. GT->DEGRADED prompt alignment: verify the COMPLETE exact prompt, physical appearance, placement and affected scope. For foreground_layer, the effect must cross subject pixels; glass/background scenery only behind the subject is a mismatch.
4. Subject visibility: decide from whole-object readability, silhouette and key parts. A transparent/semitransparent reflection, lighting, shadow, haze, droplets or dirt may cover 50% or more and still be acceptable. Pixel fraction is diagnostic, never the decision by itself. A solid-looking foreground object that dominates or hides essential parts is a solid occluder. Set borderline_visibility=true when evidence lies near the usable/unusable boundary or GT already has mild occlusion and degradation makes the result uncertain.

Use evidence arrays for concrete visible facts and issues arrays for defects. Keep each item short. Confidence values are 0..1. Enums: preexisting_occlusion/reflection_severity/effect_severity = none|light|medium|heavy; rendering_anomaly = none|mild|severe; effect_scope = foreground|background|both|global|unclear; occluder_type = none|transparent|semitransparent|solid|mixed|unclear.
If a prompt asks for a reflection, transparent overlay, illumination or shadow but IMAGE 3 creates an opaque/solid object instead, set matches_full_prompt=false even when the requested semantic content (for example a cat or person) is present. Inspect GT independently and record existing glass reflections or lighting overlap; do not assume GT is clean merely because it is called CLEAN GT.

Occlusion decision rubric:
{OCCLUSION_RUBRIC}
Return ONLY one strict JSON object, no markdown. Do not add decision/status/reject fields. Preserve pair_id exactly. Schema example: {json.dumps(schema, ensure_ascii=False)}"""
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    content.append({"type": "text", "text": f"CANDIDATE TO AUDIT {row['pair_id']}"})
    for label, key in (("IMAGE 1 SOURCE ALBEDO", "source"), ("IMAGE 2 CLEAN GT", "gt"), ("IMAGE 3 DEGRADED", "degraded")):
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": _data_url(Path(program["paths"][key]))}})
    return [{"role": "user", "content": content}]


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM response is not an object")
    return value


class QwenServer:
    def __init__(self, args):
        command = [
            str(args.qwen_python), str(args.qwen_runner), "--mode", "vl", "--server", "--single_gpu",
            "--model_path", str(args.qwen_model), "--max_new_tokens", str(args.vlm_max_new_tokens),
            "--enable_thinking", "false", "--local_files_only",
        ]
        self.process = subprocess.Popen(
            command, cwd=args.qwen_runner.parent.parent, env=os.environ.copy(), text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = args.vlm_timeout
        self._wait("__QWEN_SERVER_READY__")

    def _wait(self, prefix: str) -> str:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            for key, _ in self.selector.select(max(0.1, deadline - time.monotonic())):
                line = key.fileobj.readline()
                if line.startswith(prefix):
                    return line.strip()
                if not line and self.process.poll() is not None:
                    raise RuntimeError(f"Qwen server exited with {self.process.returncode}")
        raise TimeoutError(f"Qwen server timeout waiting for {prefix}")

    def request(self, messages) -> str:
        self.process.stdin.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self._wait("__QWEN_RESPONSE__")
        value = json.loads(line[len("__QWEN_RESPONSE__"):])
        if not value.get("ok"):
            raise RuntimeError(str(value.get("error", "unknown Qwen error")))
        return str(value["text"])

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=15)
            except Exception:
                self.process.terminate()


class OpenAICompatibleServer:
    """Stateless client for a vLLM OpenAI-compatible multimodal endpoint."""

    def __init__(self, args):
        self.url = str(args.vlm_base_url).rstrip("/") + "/chat/completions"
        self.model = str(args.vlm_served_model)
        self.timeout = args.vlm_timeout
        self.max_new_tokens = args.vlm_max_new_tokens
        self.temperature = args.vlm_temperature

    def request(self, messages) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
        choices = value.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM response has no choices: {value}")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"vLLM response content is not text: {value}")
        return content

    def close(self):
        return None


BLOCK_FIELDS = {
    "source_gt_identity": {
        "subject_consistent", "geometry_and_key_parts_preserved", "viewpoint_compatible",
        "confidence", "evidence", "issues",
    },
    "gt_quality": {
        "subject_fully_visible", "subject_severely_out_of_frame", "preexisting_occlusion",
        "reflection_severity", "rendering_anomaly", "confidence", "evidence", "issues",
    },
    "degradation_alignment": {
        "effect_present", "matches_full_prompt", "applied_to_subject", "affected_scope_respected",
        "effect_scope", "effect_severity", "confidence", "evidence", "issues",
    },
    "subject_visibility": {
        "whole_subject_recognizable", "overall_silhouette_visible", "key_parts_mostly_visible",
        "usable_for_subject_learning", "occluder_type", "solid_occluder_present",
        "estimated_occluded_fraction", "borderline_visibility", "confidence", "evidence", "issues",
    },
    "evidence_sufficiency": {
        "all_three_images_readable", "evidence_conflict", "notes", "confidence",
    },
}
BOOL_FIELDS = {
    "source_gt_identity": {"subject_consistent", "geometry_and_key_parts_preserved", "viewpoint_compatible"},
    "gt_quality": {"subject_fully_visible", "subject_severely_out_of_frame"},
    "degradation_alignment": {"effect_present", "matches_full_prompt", "applied_to_subject", "affected_scope_respected"},
    "subject_visibility": {"whole_subject_recognizable", "overall_silhouette_visible", "key_parts_mostly_visible", "usable_for_subject_learning", "solid_occluder_present", "borderline_visibility"},
    "evidence_sufficiency": {"all_three_images_readable", "evidence_conflict"},
}
ENUM_FIELDS = {
    ("gt_quality", "preexisting_occlusion"): {"none", "light", "medium", "heavy"},
    ("gt_quality", "reflection_severity"): {"none", "light", "medium", "heavy"},
    ("gt_quality", "rendering_anomaly"): {"none", "mild", "severe"},
    ("degradation_alignment", "effect_scope"): {"foreground", "background", "both", "global", "unclear"},
    ("degradation_alignment", "effect_severity"): {"none", "light", "medium", "heavy"},
    ("subject_visibility", "occluder_type"): {"none", "transparent", "semitransparent", "solid", "mixed", "unclear"},
}


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_vlm(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate the evidence schema and normalize only safe scalar aliases."""
    normalized = dict(parsed)
    problems: list[str] = []
    if not isinstance(parsed.get("pair_id"), str):
        problems.append("pair_id")
    for block_name, fields in BLOCK_FIELDS.items():
        block = parsed.get(block_name)
        if not isinstance(block, dict):
            problems.append(block_name)
            continue
        missing = sorted(fields - set(block))
        problems.extend(f"{block_name}.{name}" for name in missing)
        clean = dict(block)
        clean["confidence"] = _confidence(block.get("confidence"))
        for name in BOOL_FIELDS.get(block_name, set()):
            if isinstance(block.get(name), bool):
                clean[name] = block[name]
            else:
                problems.append(f"{block_name}.{name}:not_bool")
        for (enum_block, name), allowed in ENUM_FIELDS.items():
            if enum_block != block_name:
                continue
            value = str(block.get(name, "")).lower()
            # Qwen3-VL naturally emits scope="none" when it also reports that
            # the requested effect is absent. The schema represents that
            # observation as an unclear scope; this alias preserves the
            # negative effect finding without routing a valid response to an
            # exception-only candidate.
            if (
                block_name == "degradation_alignment"
                and name == "effect_scope"
                and value == "none"
                and block.get("effect_present") is False
            ):
                value = "unclear"
            if value not in allowed:
                problems.append(f"{block_name}.{name}:invalid_enum")
            else:
                clean[name] = value
        if block_name == "subject_visibility":
            try:
                fraction = float(block.get("estimated_occluded_fraction"))
                if not 0.0 <= fraction <= 1.0:
                    raise ValueError
                clean["estimated_occluded_fraction"] = fraction
            except (TypeError, ValueError):
                problems.append("subject_visibility.estimated_occluded_fraction:invalid")
        if "evidence" in fields:
            clean["evidence"] = [str(x) for x in block.get("evidence", [])] if isinstance(block.get("evidence"), list) else []
            clean["issues"] = [str(x) for x in block.get("issues", [])] if isinstance(block.get("issues"), list) else []
        normalized[block_name] = clean
    normalized["overall_confidence"] = _confidence(parsed.get("overall_confidence"))
    normalized["unexpected_degradations"] = (
        [str(x) for x in parsed.get("unexpected_degradations", [])]
        if isinstance(parsed.get("unexpected_degradations"), list) else []
    )
    if "overall_confidence" not in parsed:
        problems.append("overall_confidence")
    if "unexpected_degradations" not in parsed:
        problems.append("unexpected_degradations")
    return normalized, sorted(set(problems))


def _classify(parsed: dict[str, Any], row, cfg) -> tuple[str, list[str], list[str]]:
    """Map evidence to pass/reject/candidate; never trust a model decision field."""
    evidence, missing = _normalize_vlm(parsed)
    if missing:
        return "candidate", [], ["vlm_schema_missing:" + ",".join(missing)]
    if row.get("pair_id") is not None and evidence.get("pair_id") != str(row.get("pair_id")):
        return "candidate", [], ["vlm_pair_id_mismatch"]

    identity = evidence["source_gt_identity"]
    gt = evidence["gt_quality"]
    alignment = evidence["degradation_alignment"]
    visibility = evidence["subject_visibility"]
    sufficiency = evidence["evidence_sufficiency"]
    reject: list[str] = []
    candidate: list[str] = []
    internal_conflict = bool(gt["subject_fully_visible"]) and bool(gt["subject_severely_out_of_frame"])
    if internal_conflict:
        candidate.append("vlm_evidence_conflict")

    hard_findings: list[tuple[bool, str, float]] = [
        (not bool(identity["subject_consistent"]), "vlm_source_gt_object_changed", identity["confidence"]),
        (not bool(identity["geometry_and_key_parts_preserved"]), "vlm_source_gt_geometry_or_parts_changed", identity["confidence"]),
        (not bool(gt["subject_fully_visible"]), "vlm_gt_subject_not_visible_as_complete_object", gt["confidence"]),
        (bool(gt["subject_severely_out_of_frame"]), "vlm_gt_subject_severely_out_of_frame", gt["confidence"]),
        (str(gt["rendering_anomaly"]).lower() == "severe", "vlm_gt_severe_rendering_anomaly", gt["confidence"]),
        (not bool(visibility["whole_subject_recognizable"]), "vlm_subject_not_recognizable", visibility["confidence"]),
        (not bool(visibility["overall_silhouette_visible"]), "vlm_subject_overall_silhouette_hidden", visibility["confidence"]),
        (not bool(visibility["key_parts_mostly_visible"]), "vlm_subject_key_parts_hidden", visibility["confidence"]),
        (not bool(visibility["usable_for_subject_learning"]), "vlm_subject_not_usable_for_learning", visibility["confidence"]),
    ]
    for hit, reason, confidence in hard_findings:
        if not hit:
            continue
        if not internal_conflict and float(confidence) >= float(cfg["confidence_reject"]):
            reject.append(reason)
        else:
            candidate.append(reason)

    if bool(visibility["solid_occluder_present"]):
        # A solid-looking overlay is not sufficient by itself: ordinary cast
        # shadows or borderline semi-opaque reflections are commonly labeled
        # this way.  It becomes a hard semantic failure only when the model is
        # confident it violates the requested physical form and does not mark
        # the visibility judgement as borderline.
        solid_reason = "vlm_unexpected_solid_occluder"
        if (
            not internal_conflict
            and not bool(visibility["borderline_visibility"])
            and not bool(alignment["matches_full_prompt"])
            and float(visibility["confidence"]) >= float(cfg["confidence_reject"])
        ):
            reject.append(solid_reason)
        else:
            candidate.append(solid_reason)

    if not bool(alignment["effect_present"]):
        candidate.append("vlm_target_degradation_missing")
    if not bool(alignment["matches_full_prompt"]):
        candidate.append("vlm_prompt_degradation_mismatch")
    if not bool(alignment["affected_scope_respected"]):
        candidate.append("vlm_affected_scope_mismatch")
    if not bool(alignment["applied_to_subject"]) and str(row.get("degradation_type", "")) in {
        "glass_reflection", "dirty_glass", "veiling_glare", "shadow", "local_color_spill"
    }:
        candidate.append("vlm_effect_not_applied_to_subject")
    if str(alignment["effect_severity"]).lower() in {"none", "light"} and str(row.get("severity", "")).lower() == "heavy":
        candidate.append("vlm_target_effect_too_weak")
    if bool(visibility["borderline_visibility"]):
        candidate.append("vlm_subject_visibility_borderline")
    if not bool(sufficiency["all_three_images_readable"]):
        candidate.append("vlm_evidence_insufficient")
    if bool(sufficiency["evidence_conflict"]):
        candidate.append("vlm_evidence_conflict")

    confidences = [
        evidence[block]["confidence"] for block in BLOCK_FIELDS
    ] + [evidence["overall_confidence"]]
    if min(confidences) < float(cfg["confidence_review"]):
        candidate.append("vlm_low_confidence")

    return ("reject" if reject else ("candidate" if candidate else "pass")), sorted(set(reject)), sorted(set(candidate))


def run_vlm(args, thresholds) -> dict[str, int]:
    counters = {"processed": 0, "skipped": 0, "policy_bypassed": 0, "not_required": 0, "pass": 0, "candidate": 0, "reject": 0, "dependency_reject": 0}
    pending = []
    for row in stage_rows(args.manifest, args.shard_index, args.shard_count, args.limit):
        existing = read_record(args.output_root, "vlm", row["pair_id"])
        fingerprint = input_fingerprint(
            row, stage="vlm",
            extra={
                "schema": SCHEMA_VERSION, "policy": DECISION_POLICY_VERSION,
                "prompt": AUDIT_PROMPT_VERSION, "model": str(args.qwen_model),
                "temperature": args.vlm_temperature,
                "max_new_tokens": args.vlm_max_new_tokens,
                "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
            },
        )
        if (
            not getattr(args, "force", False)
            and existing
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("decision_policy_version") == DECISION_POLICY_VERSION
            and existing.get("audit_prompt_version") == AUDIT_PROMPT_VERSION
            and existing.get("input_fingerprint") == fingerprint
        ):
            counters["skipped"] += 1
            decision = existing.get("decision") or ("candidate" if existing.get("status") == "review" else existing.get("status"))
            counters[decision] = counters.get(decision, 0) + 1
            continue
        if (
            existing
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("audit_prompt_version") == AUDIT_PROMPT_VERSION
            and existing.get("model") == str(args.qwen_model)
            and float(existing.get("temperature", -1.0)) == float(args.vlm_temperature)
            and int(existing.get("max_new_tokens", -1)) == int(args.vlm_max_new_tokens)
            and isinstance(existing.get("vlm"), dict)
        ):
            # Prompt/model/image evidence is immutable across a policy-only
            # update. Reclassify the normalized evidence deterministically so
            # policy calibration does not introduce another stochastic model
            # sample.
            decision, reject, review = _classify(existing["vlm"], row, thresholds["vlm"])
            existing.update({
                "completed_at": utc_now(),
                "status": "review" if decision == "candidate" else decision,
                "decision": decision,
                "reject_reasons": reject,
                "review_reasons": review,
                "reason_codes": sorted(set(reject + review)),
                "applied_policy_rules": sorted(set(reject + review)),
                "decision_policy_version": DECISION_POLICY_VERSION,
                "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
                "input_fingerprint": fingerprint,
            })
            atomic_write_json(record_path(args.output_root, "vlm", row["pair_id"]), existing)
            counters["processed"] += 1
            counters[decision] += 1
            continue
        program = read_record(args.output_root, "programmatic", row["pair_id"])
        vision = read_record(args.output_root, "vision", row["pair_id"])
        if not program or program.get("status") == "reject":
            counters["dependency_reject"] += 1
            continue
        # SAM/DINO are strong screening signals, not sufficient semantic proof
        # of a changed object. Both review and reject-like visual flags must be
        # reconciled by the three-image VLM before a final hard reject.
        needs_vlm = bool(getattr(args, "vlm_all_pairs", False)) or str(program.get("method", "")) != "hybrid"
        if not needs_vlm:
            record = {
                "schema_version": SCHEMA_VERSION, "stage": "vlm", "completed_at": utc_now(),
                "pair_id": row["pair_id"], "status": "not_required", "reason": "deterministic_pair_clear_by_program_and_vision",
                "decision_policy_version": DECISION_POLICY_VERSION,
                "audit_prompt_version": AUDIT_PROMPT_VERSION,
                "decision": "pass", "decision_source": "hard_rule",
                "model": str(args.qwen_model), "temperature": args.vlm_temperature,
                "input_image_paths": program.get("paths", {}), "input_fingerprint": fingerprint,
            }
            atomic_write_json(record_path(args.output_root, "vlm", row["pair_id"]), record)
            counters["not_required"] += 1
            continue
        pending.append((row, program))
    if not pending:
        return counters
    backend = str(getattr(args, "vlm_backend", "local"))
    server = OpenAICompatibleServer(args) if backend == "vllm" else QwenServer(args)

    def audit_one(row, program):
        review: list[str] = []
        parsed: dict[str, Any] | None = None
        normalized: dict[str, Any] | None = None
        raw = ""
        try:
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    raw = server.request(_messages(row, program))
                    parsed = _parse_json(raw)
                    attempt_normalized, missing = _normalize_vlm(parsed)
                    if missing:
                        raise ValueError("schema missing: " + ",".join(missing))
                    normalized = attempt_normalized
                    break
                except Exception as exc:
                    normalized = None
                    last_error = exc
            if normalized is None:
                raise last_error or ValueError("empty VLM result")
            decision, reject, review = _classify(normalized, row, thresholds["vlm"])
            status = "review" if decision == "candidate" else decision
        except Exception as exc:
            decision, status, reject = "candidate", "review", []
            review = [f"vlm_exception:{type(exc).__name__}:{exc}"]
        applied_rules = sorted(set(reject + review))
        record = {
            "schema_version": SCHEMA_VERSION, "stage": "vlm", "completed_at": utc_now(),
            "pair_id": row["pair_id"], "sha256": row.get("sha256"), "task_id": row.get("task_id"),
            "status": status, "decision": decision, "decision_source": "combined_policy",
            "reject_reasons": reject, "review_reasons": review, "reason_codes": applied_rules,
            "vlm": normalized, "raw_vlm_response": raw, "raw": raw,
            "threshold_version": thresholds["schema_version"],
            "decision_policy_version": DECISION_POLICY_VERSION,
            "audit_prompt_version": AUDIT_PROMPT_VERSION,
            "model": str(args.qwen_model), "temperature": args.vlm_temperature,
            "max_new_tokens": args.vlm_max_new_tokens,
            "vlm_backend": backend,
            "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
            "input_image_paths": program.get("paths", {}),
            "input_fingerprint": input_fingerprint(
                row, stage="vlm",
                extra={
                    "schema": SCHEMA_VERSION, "policy": DECISION_POLICY_VERSION,
                    "prompt": AUDIT_PROMPT_VERSION, "model": str(args.qwen_model),
                    "temperature": args.vlm_temperature,
                    "max_new_tokens": args.vlm_max_new_tokens,
                    "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
                },
            ),
            "applied_policy_rules": applied_rules,
        }
        atomic_write_json(record_path(args.output_root, "vlm", row["pair_id"]), record)
        return decision

    try:
        concurrency = int(getattr(args, "vlm_concurrency", 1)) if backend == "vllm" else 1
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = [executor.submit(audit_one, row, program) for row, program in pending]
            for future in as_completed(futures):
                decision = future.result()
                counters["processed"] += 1
                counters[decision] += 1
    finally:
        server.close()
    return counters
