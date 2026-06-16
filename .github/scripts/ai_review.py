#!/usr/bin/env python3
"""Run AI review lanes and build structured GitHub PR reports."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


AUTHORIZED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
COMMENT_LIMIT = 60000


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--event", required=True)
    prepare.add_argument("--matrix", required=True)
    prepare.add_argument("--prompt-dir", required=True)
    prepare.add_argument("--output", required=True)

    context = sub.add_parser("context")
    context.add_argument("--repo", required=True)
    context.add_argument("--base-sha", required=True)
    context.add_argument("--head-ref", required=True)
    context.add_argument("--pr-number", required=True)
    context.add_argument("--out-dir", required=True)
    context.add_argument("--max-diff-chars", type=int, default=350000)
    context.add_argument("--max-file-chars", type=int, default=220000)

    run_lane = sub.add_parser("run-lane")
    run_lane.add_argument("--lane-json", required=True)
    run_lane.add_argument("--context", required=True)
    run_lane.add_argument("--prompt-dir", required=True)
    run_lane.add_argument("--out", required=True)

    candidates = sub.add_parser("candidates")
    candidates.add_argument("--lanes-dir", required=True)
    candidates.add_argument("--context", required=True)
    candidates.add_argument("--out-dir", required=True)
    candidates.add_argument("--output")

    verify = sub.add_parser("verify-lane")
    verify.add_argument("--lane-json", required=True)
    verify.add_argument("--context", required=True)
    verify.add_argument("--candidates", required=True)
    verify.add_argument("--prompt-dir", required=True)
    verify.add_argument("--out", required=True)

    report = sub.add_parser("report")
    report.add_argument("--lanes-dir", required=True)
    report.add_argument("--verifications-dir", required=True)
    report.add_argument("--context", required=True)
    report.add_argument("--candidates", required=True)
    report.add_argument("--out-dir", required=True)
    report.add_argument("--post-comment", action="store_true")

    args = parser.parse_args()

    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "context":
        return cmd_context(args)
    if args.command == "run-lane":
        return cmd_run_lane(args)
    if args.command == "candidates":
        return cmd_candidates(args)
    if args.command == "verify-lane":
        return cmd_verify_lane(args)
    if args.command == "report":
        return cmd_report(args)
    raise AssertionError(args.command)


def cmd_prepare(args: argparse.Namespace) -> int:
    event = read_json(pathlib.Path(args.event))
    tier, pr_number = parse_review_trigger(event)

    outputs: dict[str, Any] = {"should_run": "false"}
    if not tier or not pr_number:
        write_github_outputs(pathlib.Path(args.output), outputs)
        return 0

    matrix = read_json(pathlib.Path(args.matrix))
    if tier not in matrix:
        raise SystemExit(f"Tier {tier!r} not found in {args.matrix}")

    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    pr = github_json("GET", f"/repos/{repo}/pulls/{pr_number}", token=token)

    prompt_path = pathlib.Path(args.prompt_dir) / f"{tier}.md"
    custom_prompt = prompt_path.read_text(encoding="utf-8")
    tier_config = matrix[tier]

    outputs = {
        "should_run": "true",
        "tier": tier,
        "pr_number": str(pr_number),
        "base_sha": pr["base"]["sha"],
        "base_ref": pr["base"]["ref"],
        "head_sha": pr["head"]["sha"],
        "head_ref": f"refs/remotes/origin/pr/{pr_number}/head",
        "review_lanes": json.dumps(tier_config["review_lanes"], separators=(",", ":")),
        "verifier_lanes": json.dumps(tier_config["verifier_lanes"], separators=(",", ":")),
        "custom_prompt": custom_prompt,
    }
    write_github_outputs(pathlib.Path(args.output), outputs)
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    repo = pathlib.Path(args.repo)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = args.base_sha
    head = args.head_ref
    pr_range = f"{base}...{head}"
    diff = git_text(repo, "diff", "--find-renames", "--find-copies", "--unified=80", pr_range)
    name_status = git_text(repo, "diff", "--name-status", pr_range)
    changed_files = parse_name_status(name_status)

    diff_truncated = len(diff) > args.max_diff_chars
    if diff_truncated:
        diff = diff[: args.max_diff_chars] + "\n\n[diff truncated by ai-review]\n"

    file_context: list[dict[str, Any]] = []
    remaining = args.max_file_chars
    for changed in changed_files:
        if remaining <= 0:
            break
        if changed["status"] == "D":
            continue
        path = changed["path"]
        head_content, head_truncated = git_file_text(repo, head, path, remaining // 2)
        if head_content is not None:
            remaining -= len(head_content)
        base_content, base_truncated = git_file_text(repo, base, path, max(0, remaining // 2))
        if base_content is not None:
            remaining -= len(base_content)
        if head_content is None and base_content is None:
            continue
        file_context.append(
            {
                "path": path,
                "status": changed["status"],
                "old_path": changed.get("old_path"),
                "head": head_content,
                "head_truncated": head_truncated,
                "base": base_content,
                "base_truncated": base_truncated,
            }
        )

    context = {
        "pr_number": int(args.pr_number),
        "base_sha": base,
        "head_ref": head,
        "generated_at": int(time.time()),
        "diff_truncated": diff_truncated,
        "changed_file_count": len(changed_files),
        "changed_files": changed_files,
        "diff": diff,
        "file_context": file_context,
    }
    (out_dir / "context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    (out_dir / "pr.diff").write_text(diff, encoding="utf-8")
    return 0


def cmd_run_lane(args: argparse.Namespace) -> int:
    lane = json.loads(args.lane_json)
    context = read_json(pathlib.Path(args.context))
    prompt = load_prompt(pathlib.Path(args.prompt_dir), lane["prompt"])
    result = run_review_lane(lane, context, prompt)
    write_json(pathlib.Path(args.out), result)
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    lane_results = load_json_files(pathlib.Path(args.lanes_dir))
    context = read_json(pathlib.Path(args.context))
    candidates = build_candidates(lane_results, context)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "candidates.json", candidates)
    write_json(out_dir / "model-metrics.json", build_model_metrics(lane_results, candidates))

    if args.output:
        write_github_outputs(
            pathlib.Path(args.output),
            {
                "has_candidates": "true" if candidates["issues"] else "false",
                "candidate_count": str(len(candidates["issues"])),
            },
        )
    return 0


def cmd_verify_lane(args: argparse.Namespace) -> int:
    lane = json.loads(args.lane_json)
    context = read_json(pathlib.Path(args.context))
    candidates = read_json(pathlib.Path(args.candidates))
    prompt = load_prompt(pathlib.Path(args.prompt_dir), lane["prompt"])
    result = run_verifier_lane(lane, context, candidates, prompt)
    write_json(pathlib.Path(args.out), result)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    context = read_json(pathlib.Path(args.context))
    candidates = read_json(pathlib.Path(args.candidates))
    lane_results = load_json_files(pathlib.Path(args.lanes_dir))
    verification_results = load_json_files(pathlib.Path(args.verifications_dir))

    final = build_final_issues(candidates, verification_results)
    metrics = build_model_metrics(lane_results, candidates, verification_results)
    report = render_report(context, final, lane_results, verification_results, metrics)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "final-issues.json", final)
    write_json(out_dir / "model-metrics.json", metrics)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    if args.post_comment:
        post_or_update_comment(context["pr_number"], report, final["tier"])
    return 0


def parse_tier_command(body: str) -> str | None:
    match = re.search(r"(?im)^\s*/ai-review\s+(standard|critical)\b", body)
    if not match:
        return None
    return match.group(1).lower()


def parse_tier_label(name: str) -> str | None:
    labels = {
        "ai-review-standard": "standard",
        "ai-review-critical": "critical",
    }
    return labels.get(name.strip().lower())


def parse_review_trigger(event: dict[str, Any]) -> tuple[str | None, int | None]:
    if event.get("comment") and event.get("issue", {}).get("pull_request"):
        association = event.get("comment", {}).get("author_association", "")
        if association not in AUTHORIZED_ASSOCIATIONS:
            return None, None
        tier = parse_tier_command(event.get("comment", {}).get("body", ""))
        if not tier:
            return None, None
        return tier, int(event["issue"]["number"])

    if event.get("action") == "labeled" and event.get("pull_request"):
        tier = parse_tier_label(event.get("label", {}).get("name", ""))
        if not tier:
            return None, None
        return tier, int(event["pull_request"]["number"])

    return None, None


def run_review_lane(lane: dict[str, Any], context: dict[str, Any], prompt: str) -> dict[str, Any]:
    base_result = lane_base_result(lane, context, kind="review")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        base_result.update({"status": "skipped", "error": "OPENROUTER_API_KEY is not set"})
        return base_result

    system = textwrap.dedent(
        """\
        You are a senior code reviewer. Review only issues introduced or exposed
        by this PR. Return only valid JSON with this schema:
        {
          "summary": "brief summary",
          "findings": [
            {
              "severity": "critical|high|medium|low",
              "confidence": "high|medium|low",
              "title": "short title",
              "file": "path/to/file",
              "line": 123,
              "claim": "what is wrong",
              "evidence": "why the diff supports this",
              "suggested_fix": "specific fix"
            }
          ]
        }
        Use an empty findings array when there are no issues.
        """
    )
    user = format_review_prompt(lane, context, prompt)
    response = openrouter_chat(lane, system, user, api_key)
    base_result.update(response)
    if response["status"] != "success":
        return base_result

    parsed, parse_error = extract_json(response["raw_response"])
    findings = []
    if isinstance(parsed, dict):
        raw_findings = parsed.get("findings", [])
        if isinstance(raw_findings, list):
            findings = [normalize_finding(f, lane) for f in raw_findings if isinstance(f, dict)]
        base_result["summary"] = parsed.get("summary", "")
    elif isinstance(parsed, list):
        findings = [normalize_finding(f, lane) for f in parsed if isinstance(f, dict)]
    else:
        parse_error = parse_error or "response did not contain a JSON object"

    base_result["findings"] = [f for f in findings if f.get("claim") or f.get("title")]
    if parse_error:
        base_result["parse_error"] = parse_error
    return base_result


def run_verifier_lane(
    lane: dict[str, Any], context: dict[str, Any], candidates: dict[str, Any], prompt: str
) -> dict[str, Any]:
    base_result = lane_base_result(lane, context, kind="verification")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        base_result.update({"status": "skipped", "error": "OPENROUTER_API_KEY is not set"})
        return base_result
    if not candidates.get("issues"):
        base_result.update({"status": "skipped", "error": "No candidate issues to verify"})
        return base_result

    system = textwrap.dedent(
        """\
        You verify AI code review findings. Do not create new findings. Return
        only valid JSON with this schema:
        {
          "summary": "brief summary",
          "verifications": [
            {
              "issue_id": "AI-001",
              "status": "confirmed|rejected|uncertain",
              "confidence": "high|medium|low",
              "rationale": "why"
            }
          ]
        }
        """
    )
    user = format_verification_prompt(lane, context, candidates, prompt)
    response = openrouter_chat(lane, system, user, api_key)
    base_result.update(response)
    if response["status"] != "success":
        return base_result

    parsed, parse_error = extract_json(response["raw_response"])
    verifications = []
    if isinstance(parsed, dict):
        raw_items = parsed.get("verifications", [])
        if isinstance(raw_items, list):
            verifications = [normalize_verification(v, lane) for v in raw_items if isinstance(v, dict)]
        base_result["summary"] = parsed.get("summary", "")
    elif isinstance(parsed, list):
        verifications = [normalize_verification(v, lane) for v in parsed if isinstance(v, dict)]
    else:
        parse_error = parse_error or "response did not contain a JSON object"
    base_result["verifications"] = [v for v in verifications if v.get("issue_id")]
    if parse_error:
        base_result["parse_error"] = parse_error
    return base_result


def lane_base_result(lane: dict[str, Any], context: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": "success",
        "tier": lane.get("tier") or infer_tier_from_lane(lane),
        "pr_number": context["pr_number"],
        "lane_id": lane["id"],
        "model": lane["model"],
        "prompt": lane["prompt"],
        "findings": [],
        "verifications": [],
    }


def infer_tier_from_lane(lane: dict[str, Any]) -> str:
    lane_id = lane.get("id", "")
    prompt = lane.get("prompt", "")
    if "critical" in lane_id or prompt == "critical" or "critical" in prompt:
        return "critical"
    return "standard"


def openrouter_chat(lane: dict[str, Any], system: str, user: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": lane["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": lane.get("temperature", 0.1),
        "max_tokens": lane.get("max_output_tokens", 2400),
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": github_repo_url(),
        "X-Title": "lambda_vm AI Review",
    }
    req = urllib.request.Request(OPENROUTER_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"status": "error", "error": f"OpenRouter HTTP {exc.code}: {body[:1000]}"}
    except Exception as exc:
        return {"status": "error", "error": f"OpenRouter request failed: {exc}"}

    try:
        content = parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {"status": "error", "error": f"Unexpected OpenRouter response: {json.dumps(parsed)[:1000]}"}

    return {
        "status": "success",
        "raw_response": content,
        "usage": parsed.get("usage", {}),
        "openrouter_id": parsed.get("id"),
    }


def format_review_prompt(lane: dict[str, Any], context: dict[str, Any], prompt: str) -> str:
    return "\n\n".join(
        [
            f"PR #{context['pr_number']}",
            f"Lane: {lane['id']}",
            f"Model: {lane['model']}",
            "Lane instructions:\n" + prompt.strip(),
            format_changed_files(context),
            "Diff:\n" + context.get("diff", ""),
            format_file_context(context),
        ]
    )


def format_verification_prompt(
    lane: dict[str, Any], context: dict[str, Any], candidates: dict[str, Any], prompt: str
) -> str:
    compact_candidates = [
        {
            "issue_id": issue["issue_id"],
            "severity": issue["severity"],
            "title": issue["title"],
            "file": issue.get("file"),
            "line": issue.get("line"),
            "claim": issue["claim"],
            "evidence": issue.get("evidence"),
            "found_by": issue["found_by"],
        }
        for issue in candidates.get("issues", [])
    ]
    return "\n\n".join(
        [
            f"PR #{context['pr_number']}",
            f"Verifier lane: {lane['id']}",
            "Verifier instructions:\n" + prompt.strip(),
            "Candidate findings:\n" + json.dumps(compact_candidates, indent=2),
            format_changed_files(context),
            "Diff:\n" + context.get("diff", ""),
            format_file_context(context),
        ]
    )


def format_changed_files(context: dict[str, Any]) -> str:
    lines = [f"Changed files ({context.get('changed_file_count', 0)}):"]
    for item in context.get("changed_files", []):
        old_path = f" from {item['old_path']}" if item.get("old_path") else ""
        lines.append(f"- {item['status']} {item['path']}{old_path}")
    if context.get("diff_truncated"):
        lines.append("- Warning: diff was truncated by ai-review.")
    return "\n".join(lines)


def format_file_context(context: dict[str, Any]) -> str:
    parts = ["Changed file context:"]
    for item in context.get("file_context", []):
        parts.append(f"--- {item['path']} ({item['status']}) HEAD ---")
        if item.get("head") is None:
            parts.append("[not available]")
        else:
            suffix = "\n[head content truncated]" if item.get("head_truncated") else ""
            parts.append(item["head"] + suffix)
        if item.get("base") is not None:
            parts.append(f"--- {item['path']} BASE ---")
            suffix = "\n[base content truncated]" if item.get("base_truncated") else ""
            parts.append(item["base"] + suffix)
    return "\n".join(parts)


def build_candidates(lane_results: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    all_findings = []
    tier = "standard"
    for result in lane_results:
        tier = result.get("tier") or tier
        if result.get("kind") != "review" or result.get("status") != "success":
            continue
        for finding in result.get("findings", []):
            normalized = normalize_finding(finding, result)
            normalized["source_lane"] = result["lane_id"]
            normalized["source_model"] = result["model"]
            normalized["source_prompt"] = result["prompt"]
            all_findings.append(normalized)

    for finding in sorted(all_findings, key=finding_sort_key):
        group = find_duplicate_group(groups, finding)
        if group is None:
            issue_id = f"AI-{len(groups) + 1:03d}"
            group = {
                "issue_id": issue_id,
                "status": "candidate",
                "severity": finding["severity"],
                "title": finding["title"],
                "file": finding.get("file"),
                "line": finding.get("line"),
                "claim": finding["claim"],
                "evidence": finding.get("evidence", ""),
                "suggested_fix": finding.get("suggested_fix", ""),
                "found_by": [],
                "sources": [],
            }
            groups.append(group)
        merge_finding_into_group(group, finding)

    return {
        "tier": tier,
        "pr_number": context["pr_number"],
        "base_sha": context["base_sha"],
        "generated_at": int(time.time()),
        "issues": groups,
    }


def find_duplicate_group(groups: list[dict[str, Any]], finding: dict[str, Any]) -> dict[str, Any] | None:
    for group in groups:
        if finding.get("file") and group.get("file") and finding["file"] != group["file"]:
            continue
        same_line = False
        if finding.get("line") is not None and group.get("line") is not None:
            same_line = abs(int(finding["line"]) - int(group["line"])) <= 8
        text_score = similarity(group.get("claim", "") + " " + group.get("title", ""), finding.get("claim", "") + " " + finding.get("title", ""))
        if same_line and text_score >= 0.45:
            return group
        if text_score >= 0.72:
            return group
    return None


def merge_finding_into_group(group: dict[str, Any], finding: dict[str, Any]) -> None:
    source = f"{finding['source_lane']}:{finding['source_model']}"
    if source not in group["found_by"]:
        group["found_by"].append(source)
    group["sources"].append(
        {
            "lane_id": finding["source_lane"],
            "model": finding["source_model"],
            "prompt": finding["source_prompt"],
            "severity": finding["severity"],
            "confidence": finding.get("confidence"),
            "title": finding.get("title"),
            "claim": finding.get("claim"),
            "evidence": finding.get("evidence"),
            "suggested_fix": finding.get("suggested_fix"),
        }
    )
    group["severity"] = higher_severity(group["severity"], finding["severity"])
    if not group.get("evidence") and finding.get("evidence"):
        group["evidence"] = finding["evidence"]
    if not group.get("suggested_fix") and finding.get("suggested_fix"):
        group["suggested_fix"] = finding["suggested_fix"]


def build_final_issues(candidates: dict[str, Any], verification_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_issue: dict[str, list[dict[str, Any]]] = {}
    for result in verification_results:
        if result.get("kind") != "verification" or result.get("status") != "success":
            continue
        for item in result.get("verifications", []):
            by_issue.setdefault(item["issue_id"], []).append(item)

    final_issues = []
    for issue in candidates.get("issues", []):
        verifications = by_issue.get(issue["issue_id"], [])
        confirmed_by = [v["verifier"] for v in verifications if v["status"] == "confirmed"]
        rejected_by = [v["verifier"] for v in verifications if v["status"] == "rejected"]
        uncertain_by = [v["verifier"] for v in verifications if v["status"] == "uncertain"]
        status = "candidate"
        if confirmed_by:
            status = "confirmed"
        elif rejected_by and not uncertain_by:
            status = "rejected"
        elif uncertain_by:
            status = "uncertain"

        final_issue = dict(issue)
        final_issue.update(
            {
                "status": status,
                "verified_by": confirmed_by,
                "rejected_by": rejected_by,
                "uncertain_by": uncertain_by,
                "verification": verifications,
            }
        )
        final_issues.append(final_issue)

    return {
        "tier": candidates.get("tier", "standard"),
        "pr_number": candidates["pr_number"],
        "base_sha": candidates["base_sha"],
        "generated_at": int(time.time()),
        "issues": final_issues,
    }


def render_report(
    context: dict[str, Any],
    final: dict[str, Any],
    lane_results: list[dict[str, Any]],
    verification_results: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> str:
    tier = final["tier"]
    marker = f"<!-- ai-review:{tier} -->"
    visible_issues = [i for i in final["issues"] if i["status"] != "rejected"]
    rejected = [i for i in final["issues"] if i["status"] == "rejected"]
    lines = [
        marker,
        f"## AI Review ({tier})",
        "",
        f"PR #{context['pr_number']} · {len(context.get('changed_files', []))} changed files",
    ]
    if context.get("diff_truncated"):
        lines.append("")
        lines.append("> Warning: the diff was truncated before review.")

    lines.extend(["", "### Findings", ""])
    if visible_issues:
        lines.append("| Status | Sev | Location | Finding | Found by | Verified by |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for issue in visible_issues[:20]:
            lines.append(
                "| {status} | {severity} | {where} | {finding} | {found_by} | {verified_by} |".format(
                    status=issue["status"],
                    severity=issue["severity"],
                    where=md_escape(format_location(issue)),
                    finding=md_escape(issue["title"] or issue["claim"]),
                    found_by=md_escape(", ".join(issue.get("found_by", []))),
                    verified_by=md_escape(", ".join(issue.get("verified_by", [])) or "-"),
                )
            )
        if len(visible_issues) > 20:
            lines.append(f"\n_Only the first 20 findings are shown. See artifacts for all {len(visible_issues)}._")
    else:
        lines.append("No non-rejected structured findings were reported.")

    for issue in visible_issues[:10]:
        lines.extend(
            [
                "",
                f"<details><summary>{md_escape(issue['issue_id'])}: {md_escape(issue['title'] or issue['claim'])}</summary>",
                "",
                f"- Status: `{issue['status']}`",
                f"- Severity: `{issue['severity']}`",
                f"- Location: `{format_location(issue)}`",
                f"- Found by: `{', '.join(issue.get('found_by', []))}`",
                f"- Verified by: `{', '.join(issue.get('verified_by', [])) or '-'}`",
                f"- Rejected by: `{', '.join(issue.get('rejected_by', [])) or '-'}`",
                "",
                "**Claim**",
                "",
                issue.get("claim", "").strip() or "-",
                "",
                "**Evidence**",
                "",
                issue.get("evidence", "").strip() or "-",
                "",
                "**Suggested fix**",
                "",
                issue.get("suggested_fix", "").strip() or "-",
                "",
                "</details>",
            ]
        )

    lines.extend(["", "### Reviewer Lanes", ""])
    lines.append("| Lane | Model | Prompt | Status | Findings |")
    lines.append("| --- | --- | --- | --- | ---: |")
    for lane in sorted((r for r in lane_results if r.get("kind") == "review"), key=lambda r: r.get("lane_id", "")):
        lines.append(
            "| {lane} | {model} | {prompt} | {status} | {count} |".format(
                lane=md_escape(lane.get("lane_id", "")),
                model=md_escape(lane.get("model", "")),
                prompt=md_escape(lane.get("prompt", "")),
                status=md_escape(lane_status(lane)),
                count=len(lane.get("findings", [])),
            )
        )

    if verification_results:
        lines.extend(["", "### Verification Lanes", ""])
        lines.append("| Lane | Model | Status | Confirmed | Rejected | Uncertain |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for lane in sorted(verification_results, key=lambda r: r.get("lane_id", "")):
            counts = verification_counts(lane)
            lines.append(
                "| {lane} | {model} | {status} | {confirmed} | {rejected} | {uncertain} |".format(
                    lane=md_escape(lane.get("lane_id", "")),
                    model=md_escape(lane.get("model", "")),
                    status=md_escape(lane_status(lane)),
                    confirmed=counts["confirmed"],
                    rejected=counts["rejected"],
                    uncertain=counts["uncertain"],
                )
            )

    if tier == "critical":
        lines.extend(
            [
                "",
                "Native Codex and Claude critical reviews are triggered as separate reviewer comments. "
                "They are not included in this structured provenance report yet.",
            ]
        )
    if rejected:
        lines.append(f"\nRejected candidates: {len(rejected)}. See `final-issues.json` artifact for details.")
    lines.append("\nRaw lane outputs, candidates, final issues, and model metrics are uploaded as workflow artifacts.")

    rendered = "\n".join(lines)
    if len(rendered) > COMMENT_LIMIT:
        rendered = rendered[: COMMENT_LIMIT - 200] + "\n\n[comment truncated; see workflow artifacts]\n"
    return rendered


def build_model_metrics(
    lane_results: list[dict[str, Any]],
    candidates: dict[str, Any],
    verification_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "generated_at": int(time.time()),
        "lanes": {},
    }
    for result in lane_results:
        lane_id = result.get("lane_id")
        if not lane_id:
            continue
        metrics["lanes"][lane_id] = {
            "kind": result.get("kind"),
            "model": result.get("model"),
            "prompt": result.get("prompt"),
            "status": result.get("status"),
            "findings": len(result.get("findings", [])),
            "parse_error": result.get("parse_error"),
            "error": result.get("error"),
            "usage": result.get("usage", {}),
            "unique_candidates_found": 0,
        }

    for issue in candidates.get("issues", []):
        lanes = {source.get("lane_id") for source in issue.get("sources", [])}
        for lane_id in lanes:
            if lane_id in metrics["lanes"]:
                metrics["lanes"][lane_id]["unique_candidates_found"] += 1

    if verification_results is not None:
        metrics["verification_lanes"] = {}
        for result in verification_results:
            lane_id = result.get("lane_id")
            if not lane_id:
                continue
            metrics["verification_lanes"][lane_id] = {
                "model": result.get("model"),
                "prompt": result.get("prompt"),
                "status": result.get("status"),
                "verifications": len(result.get("verifications", [])),
                "counts": verification_counts(result),
                "parse_error": result.get("parse_error"),
                "error": result.get("error"),
                "usage": result.get("usage", {}),
            }
    return metrics


def normalize_finding(item: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    severity = normalize_severity(item.get("severity", "medium"))
    line = item.get("line")
    try:
        line = int(line) if line not in (None, "") else None
    except (TypeError, ValueError):
        line = None
    title = str(item.get("title") or item.get("summary") or item.get("claim") or "").strip()
    claim = str(item.get("claim") or item.get("description") or title).strip()
    return {
        "severity": severity,
        "confidence": normalize_confidence(item.get("confidence", "medium")),
        "title": title[:180],
        "file": clean_path(item.get("file") or item.get("path")),
        "line": line,
        "claim": claim,
        "evidence": str(item.get("evidence") or item.get("why") or "").strip(),
        "suggested_fix": str(item.get("suggested_fix") or item.get("fix") or "").strip(),
        "source_lane": item.get("source_lane") or source.get("lane_id", ""),
        "source_model": item.get("source_model") or source.get("model", ""),
        "source_prompt": item.get("source_prompt") or source.get("prompt", ""),
    }


def normalize_verification(item: dict[str, Any], lane: dict[str, Any]) -> dict[str, Any]:
    status = str(item.get("status", "uncertain")).strip().lower()
    if status not in {"confirmed", "rejected", "uncertain"}:
        status = "uncertain"
    return {
        "issue_id": str(item.get("issue_id") or item.get("id") or "").strip(),
        "status": status,
        "confidence": normalize_confidence(item.get("confidence", "medium")),
        "rationale": str(item.get("rationale") or item.get("reason") or "").strip(),
        "verifier": f"{lane['id']}:{lane['model']}",
        "lane_id": lane["id"],
        "model": lane["model"],
    }


def parse_name_status(text: str) -> list[dict[str, Any]]:
    changed = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") or status.startswith("C"):
            changed.append({"status": status[0], "old_path": parts[1], "path": parts[2]})
        else:
            changed.append({"status": status[0], "path": parts[-1]})
    return changed


def git_text(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace")


def git_file_text(repo: pathlib.Path, ref: str, path: str, max_chars: int) -> tuple[str | None, bool]:
    if max_chars <= 0:
        return "", True
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{ref}:{path}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None, False
    if b"\x00" in result.stdout[:4096]:
        return "[binary file omitted]", False
    text = result.stdout.decode("utf-8", errors="replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return text, truncated


def load_prompt(prompt_dir: pathlib.Path, prompt_id: str) -> str:
    candidates = [
        prompt_dir / f"{prompt_id}.md",
        prompt_dir / "lanes" / f"{prompt_id}.md",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise SystemExit(f"Prompt {prompt_id!r} not found under {prompt_dir}")


def load_json_files(root: pathlib.Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    results = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and ("lane_id" in data or "issues" in data):
            results.append(data)
    return results


def extract_json(text: str) -> tuple[Any, str | None]:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        try:
            return json.loads(block), None
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            return parsed, None
        except json.JSONDecodeError:
            continue
    return None, "could not parse JSON from model response"


def github_json(method: str, path: str, token: str, body: dict[str, Any] | None = None) -> Any:
    url = f"https://api.github.com{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else None


def post_or_update_comment(pr_number: int, body: str, tier: str) -> None:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    marker = f"<!-- ai-review:{tier} -->"
    comments = github_json("GET", f"/repos/{repo}/issues/{pr_number}/comments?per_page=100", token=token)
    existing_id = None
    for comment in reversed(comments):
        if marker in comment.get("body", ""):
            existing_id = comment["id"]
            break
    if existing_id:
        github_json("PATCH", f"/repos/{repo}/issues/comments/{existing_id}", token=token, body={"body": body})
    else:
        github_json("POST", f"/repos/{repo}/issues/{pr_number}/comments", token=token, body={"body": body})


def write_github_outputs(path: pathlib.Path, outputs: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in outputs.items():
            text = str(value)
            if "\n" in text:
                delimiter = f"__AI_REVIEW_{key.upper()}__"
                handle.write(f"{key}<<{delimiter}\n{text}\n{delimiter}\n")
            else:
                handle.write(f"{key}={text}\n")


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def normalize_severity(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"critical", "high", "medium", "low"}:
        return text
    if text in {"med", "moderate"}:
        return "medium"
    return "medium"


def normalize_confidence(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    return "medium"


def clean_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "none", "-"}:
        return None
    return text


def severity_rank(severity: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(severity, 2)


def higher_severity(left: str, right: str) -> str:
    return left if severity_rank(left) <= severity_rank(right) else right


def finding_sort_key(finding: dict[str, Any]) -> tuple[int, str, int]:
    line = finding.get("line")
    return (severity_rank(finding["severity"]), finding.get("file") or "", int(line) if line is not None else 0)


def similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    return difflib.SequenceMatcher(None, left_norm, right_norm).ratio()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def format_location(issue: dict[str, Any]) -> str:
    file = issue.get("file") or "unknown"
    line = issue.get("line")
    return f"{file}:{line}" if line is not None else file


def md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def lane_status(lane: dict[str, Any]) -> str:
    status = lane.get("status", "unknown")
    if status in {"error", "skipped"} and lane.get("error"):
        return f"{status}: {lane['error'][:120]}"
    if lane.get("parse_error"):
        return f"{status}: parse warning"
    return status


def verification_counts(result: dict[str, Any]) -> dict[str, int]:
    counts = {"confirmed": 0, "rejected": 0, "uncertain": 0}
    for item in result.get("verifications", []):
        status = item.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def github_repo_url() -> str:
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return f"https://github.com/{repo}"
    return "https://github.com/yetanotherco/lambda_vm"


if __name__ == "__main__":
    raise SystemExit(main())
