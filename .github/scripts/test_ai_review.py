#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import unittest
from typing import Any


SCRIPT_PATH = pathlib.Path(__file__).with_name("ai_review.py")


def load_ai_review() -> Any:
    spec = importlib.util.spec_from_file_location("ai_review", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ai_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ai_review = load_ai_review()


class AiReviewParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lane = {
            "id": "mimo-tests",
            "model": "xiaomi/mimo-v2.5",
            "prompt": "tests",
        }
        self.context = {
            "pr_number": 671,
            "base_sha": "base",
            "changed_files": [],
            "diff": "",
            "file_context": [],
        }
        self.original_openrouter_chat = ai_review.openrouter_chat
        self.original_repair_json = ai_review.repair_json
        self.original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        ai_review.openrouter_chat = self.original_openrouter_chat
        ai_review.repair_json = self.original_repair_json
        if self.original_api_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self.original_api_key

    def test_extract_json_rejects_malformed_fenced_json_when_repair_unavailable(self) -> None:
        ai_review.repair_json = None
        raw_response = '''```json
{
  "summary": "tests",
  "findings": [
    {
      "severity": "low",
      "confidence": "high",
      "title": "Missing tests",
      "claim": "The script has no parser tests.",
      "suggested_fix": "Add tests for:
1. malformed JSON
2. empty responses"
    }
  ]
}
```'''

        parsed, parse_error = ai_review.extract_json(raw_response, required_key="findings")

        self.assertIsNone(parsed)
        self.assertIn("invalid JSON in fenced block", parse_error)

    def test_extract_json_recovers_malformed_json_via_repair(self) -> None:
        recovered = {"summary": "tests", "findings": [{"title": "Missing tests"}]}
        ai_review.repair_json = lambda candidate, return_objects=False: recovered

        # Unescaped inner quotes that strict json.loads cannot parse.
        raw_response = '```json\n{"findings": [{"title": "uses contains("a", "b")"}]}\n```'
        parsed, parse_error = ai_review.extract_json(raw_response, required_key="findings")

        self.assertEqual(parsed, recovered)
        self.assertIn("recovered malformed JSON via json-repair", parse_error)

class AiReviewExtractorTests(unittest.TestCase):
    def test_openrouter_payload_omits_json_mode_and_reasoning_by_default(self) -> None:
        lane = {
            "id": "glm-standard",
            "model": "z-ai/glm-5.1",
            "prompt": "standard",
            "max_output_tokens": 32000,
        }

        payload = ai_review.openrouter_payload(lane, "system", "user")

        # Forcing json_object mode makes reasoning models reason until truncated
        # without emitting content, so it must not be sent unless a lane opts in.
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["max_tokens"], 32000)
        self.assertNotIn("reasoning", payload)

    def test_openrouter_payload_passes_through_explicit_response_format(self) -> None:
        lane = {
            "id": "glm-standard",
            "model": "z-ai/glm-5.1",
            "prompt": "standard",
            "response_format": {"type": "json_object"},
        }

        payload = ai_review.openrouter_payload(lane, "system", "user")

        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_strip_sse_comments_drops_keepalive_and_whitespace(self) -> None:
        body = ": OPENROUTER PROCESSING\n: OPENROUTER PROCESSING\n{\"findings\": []}\n"
        self.assertEqual(ai_review.strip_sse_comments(body), '{"findings": []}')
        # whitespace/keepalive-only body collapses to empty (the transient failure case)
        self.assertEqual(ai_review.strip_sse_comments("\n\n   \n"), "")

    def test_openrouter_chat_retries_on_empty_body(self) -> None:
        good = json.dumps(
            {"choices": [{"message": {"content": '{"findings": []}'}, "finish_reason": "stop"}],
             "provider": "Novita", "usage": {}, "id": "gen-1"}
        )
        bodies = iter(["\n\n   \n", good])  # whitespace-only body, then valid JSON

        class FakeResp:
            def __init__(self, text: str) -> None:
                self._b = text.encode("utf-8")

            def __enter__(self) -> "FakeResp":
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

            def read(self) -> bytes:
                return self._b

        calls = {"n": 0}

        def fake_urlopen(req: Any, timeout: Any = None) -> "FakeResp":
            calls["n"] += 1
            return FakeResp(next(bodies))

        original_urlopen = ai_review.urllib.request.urlopen
        original_sleep = ai_review.time.sleep
        ai_review.urllib.request.urlopen = fake_urlopen
        ai_review.time.sleep = lambda *a, **k: None
        try:
            result = ai_review.openrouter_chat({"model": "minimax/minimax-m3"}, "sys", "usr", "key")
        finally:
            ai_review.urllib.request.urlopen = original_urlopen
            ai_review.time.sleep = original_sleep

        self.assertEqual(calls["n"], 2)  # retried once after the empty body
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["provider"], "Novita")

    def test_opencode_assistant_text_extracts_text_events(self) -> None:
        stream = "\n".join(
            [
                json.dumps({"type": "step_start"}),
                json.dumps({"type": "tool_use", "part": {"tool": "read"}}),
                json.dumps({"type": "text", "part": {"text": "let me look..."}}),
                json.dumps({"type": "text", "part": {"text": '{"summary":"s","findings":[]}'}}),
                "not-json-noise",
            ]
        )
        text = ai_review.opencode_assistant_text(stream)
        parsed, parse_error = ai_review.extract_json(text, required_key="findings")
        self.assertIsNone(parse_error)
        self.assertEqual(parsed, {"summary": "s", "findings": []})

    def test_extract_json_accepts_bare_json(self) -> None:
        parsed, parse_error = ai_review.extract_json('{"summary":"ok","findings":[]}', required_key="findings")

        self.assertIsNone(parse_error)
        self.assertEqual(parsed, {"summary": "ok", "findings": []})

    def test_extract_json_falls_back_to_later_valid_fenced_block(self) -> None:
        raw_response = """First try:
```json
{"findings": [
```

Second try:
```json
{"summary": "ok", "findings": []}
```"""

        parsed, parse_error = ai_review.extract_json(raw_response, required_key="findings")

        self.assertIsNone(parse_error)
        self.assertEqual(parsed, {"summary": "ok", "findings": []})

    def test_extract_json_rejects_wrong_top_level_shape(self) -> None:
        raw_response = """```json
{"severity": "low", "claim": "Nested finding object only"}
```"""

        parsed, parse_error = ai_review.extract_json(raw_response, required_key="findings")

        self.assertIsNone(parsed)
        self.assertIn("top-level object with 'findings'", parse_error)


class AiReviewTriggerTests(unittest.TestCase):
    def test_authorized_comment_trigger_returns_tier_and_pr_number(self) -> None:
        event = {
            "comment": {
                "author_association": "MEMBER",
                "body": "please run\n/ai-review Critical\nthanks",
            },
            "issue": {
                "number": 671,
                "pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/671"},
            },
        }

        self.assertEqual(ai_review.parse_review_trigger(event), ("critical", 671))

    def test_unauthorized_comment_trigger_is_ignored(self) -> None:
        event = {
            "comment": {
                "author_association": "CONTRIBUTOR",
                "body": "/ai-review standard",
            },
            "issue": {
                "number": 671,
                "pull_request": {"url": "https://api.github.com/repos/org/repo/pulls/671"},
            },
        }

        self.assertEqual(ai_review.parse_review_trigger(event), (None, None))

    def test_label_trigger_maps_to_tier(self) -> None:
        event = {
            "action": "labeled",
            "label": {"name": "AI-Review-Critical"},
            "pull_request": {"number": 671},
        }

        self.assertEqual(ai_review.parse_review_trigger(event), ("critical", 671))

    def test_same_repo_pr_is_not_a_fork(self) -> None:
        pr = {
            "head": {"repo": {"full_name": "org/repo"}},
            "base": {"repo": {"full_name": "org/repo"}},
        }
        self.assertFalse(ai_review.pr_is_from_fork(pr))

    def test_fork_pr_is_detected(self) -> None:
        pr = {
            "head": {"repo": {"full_name": "attacker/repo"}},
            "base": {"repo": {"full_name": "org/repo"}},
        }
        self.assertTrue(ai_review.pr_is_from_fork(pr))

    def test_deleted_fork_repo_is_treated_as_fork(self) -> None:
        # head.repo is null when the fork was deleted; must not be treated as same-repo
        pr = {"head": {"repo": None}, "base": {"repo": {"full_name": "org/repo"}}}
        self.assertTrue(ai_review.pr_is_from_fork(pr))

    def test_safe_lane_ids_are_accepted(self) -> None:
        for lane_id in ("glm", "deepseek-verifier", "lane_1.2", "GPT-5"):
            ai_review.assert_safe_lane_id(lane_id)  # must not raise

    def test_unsafe_lane_ids_are_rejected(self) -> None:
        for lane_id in ("a;b", "$(curl evil)", "a b", "`id`", "", "x/../y"):
            with self.assertRaises(SystemExit):
                ai_review.assert_safe_lane_id(lane_id)


class AiReviewCandidateTests(unittest.TestCase):
    def test_build_candidates_merges_duplicate_findings_and_preserves_sources(self) -> None:
        context = {"pr_number": 671, "base_sha": "base"}
        lane_results = [
            {
                "kind": "review",
                "status": "success",
                "tier": "standard",
                "lane_id": "lane-a",
                "model": "model-a",
                "prompt": "correctness",
                "findings": [
                    {
                        "severity": "medium",
                        "confidence": "high",
                        "title": "Parser accepts malformed output",
                        "file": ".github/scripts/ai_review.py",
                        "line": 100,
                        "claim": "The parser can treat malformed model output as a clean result.",
                        "evidence": "Malformed fenced JSON is salvaged from a nested object.",
                        "suggested_fix": "Require the top-level findings wrapper.",
                    }
                ],
            },
            {
                "kind": "review",
                "status": "success",
                "tier": "standard",
                "lane_id": "lane-b",
                "model": "model-b",
                "prompt": "tests",
                "findings": [
                    {
                        "severity": "high",
                        "confidence": "medium",
                        "title": "Malformed output can be accepted",
                        "file": ".github/scripts/ai_review.py",
                        "line": 104,
                        "claim": "Malformed model output can be treated as a successful empty result.",
                        "evidence": "The parsed object may not contain the findings wrapper.",
                        "suggested_fix": "Keep malformed JSON as a parse warning.",
                    },
                    {
                        "severity": "medium",
                        "confidence": "medium",
                        "title": "Parser accepts malformed output",
                        "file": "docs/ai-review.md",
                        "line": 100,
                        "claim": "The parser can treat malformed model output as a clean result.",
                        "evidence": "Same claim in a different file should not merge.",
                        "suggested_fix": "Keep separate locations separate.",
                    },
                ],
            },
        ]

        candidates = ai_review.build_candidates(lane_results, context)

        self.assertEqual(len(candidates["issues"]), 2)
        script_issue = next(issue for issue in candidates["issues"] if issue["file"] == ".github/scripts/ai_review.py")
        docs_issue = next(issue for issue in candidates["issues"] if issue["file"] == "docs/ai-review.md")
        self.assertEqual(script_issue["severity"], "high")
        self.assertEqual(set(script_issue["found_by"]), {"lane-a:model-a", "lane-b:model-b"})
        self.assertEqual(len(script_issue["sources"]), 2)
        self.assertEqual(docs_issue["found_by"], ["lane-b:model-b"])


class AiReviewVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lane = {
            "id": "qwen-standard-verifier",
            "model": "qwen/qwen3.7-plus",
            "prompt": "verify",
        }
        self.context = {
            "pr_number": 671,
            "base_sha": "base",
            "changed_files": [],
            "diff": "",
            "file_context": [],
        }
        self.candidates = {
            "tier": "standard",
            "pr_number": 671,
            "base_sha": "base",
            "issues": [
                {
                    "issue_id": "AI-001",
                    "severity": "medium",
                    "title": "Parser issue",
                    "file": ".github/scripts/ai_review.py",
                    "line": 1,
                    "claim": "Parser can misclassify output.",
                    "evidence": "Malformed JSON case.",
                    "found_by": ["lane-a:model-a"],
                }
            ],
        }
        self.original_openrouter_chat = ai_review.openrouter_chat
        self.original_repair_json = ai_review.repair_json
        self.original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        ai_review.openrouter_chat = self.original_openrouter_chat
        ai_review.repair_json = self.original_repair_json
        if self.original_api_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self.original_api_key

    def test_build_final_issues_applies_verification_statuses(self) -> None:
        candidates = {
            "tier": "standard",
            "pr_number": 671,
            "base_sha": "base",
            "issues": [
                {"issue_id": "AI-001", "severity": "high", "title": "A", "claim": "A", "found_by": []},
                {"issue_id": "AI-002", "severity": "medium", "title": "B", "claim": "B", "found_by": []},
                {"issue_id": "AI-003", "severity": "low", "title": "C", "claim": "C", "found_by": []},
                {"issue_id": "AI-004", "severity": "low", "title": "D", "claim": "D", "found_by": []},
                {"issue_id": "AI-005", "severity": "high", "title": "E", "claim": "E", "found_by": []},
            ],
        }
        verification_results = [
            {
                "kind": "verification",
                "status": "success",
                "verifications": [
                    {
                        "issue_id": "AI-001",
                        "status": "confirmed",
                        "verifier": "verifier-a:model",
                    },
                    {
                        "issue_id": "AI-002",
                        "status": "rejected",
                        "verifier": "verifier-a:model",
                    },
                    {
                        "issue_id": "AI-003",
                        "status": "uncertain",
                        "verifier": "verifier-b:model",
                    },
                    {
                        "issue_id": "AI-005",
                        "status": "confirmed",
                        "verifier": "verifier-a:model",
                    },
                    {
                        "issue_id": "AI-005",
                        "status": "rejected",
                        "verifier": "verifier-b:model",
                    },
                ],
            }
        ]

        final = ai_review.build_final_issues(candidates, verification_results)
        by_id = {issue["issue_id"]: issue for issue in final["issues"]}

        self.assertEqual(by_id["AI-001"]["status"], "confirmed")
        self.assertEqual(by_id["AI-001"]["verified_by"], ["verifier-a:model"])
        self.assertEqual(by_id["AI-002"]["status"], "rejected")
        self.assertEqual(by_id["AI-002"]["rejected_by"], ["verifier-a:model"])
        self.assertEqual(by_id["AI-003"]["status"], "uncertain")
        self.assertEqual(by_id["AI-003"]["uncertain_by"], ["verifier-b:model"])
        self.assertEqual(by_id["AI-004"]["status"], "candidate")
        # conflicting verifiers (one confirms, one rejects) must surface as uncertain
        self.assertEqual(by_id["AI-005"]["status"], "uncertain")


class AiReviewSubmissionTests(unittest.TestCase):
    def _write(self, content: str) -> pathlib.Path:
        import tempfile

        path = pathlib.Path(tempfile.mkdtemp()) / "sub.json"
        path.write_text(content, encoding="utf-8")
        return path

    def test_read_submission_placeholder_not_submitted(self) -> None:
        path = self._write(json.dumps({"submitted": False, "findings": [], "summary": ""}))
        sub = ai_review.read_submission(path)
        self.assertFalse(sub["submitted"])
        self.assertEqual(sub["items"], [])

    def test_read_submission_submitted_with_findings(self) -> None:
        path = self._write(
            json.dumps({"submitted": True, "summary": "s", "findings": [{"title": "t", "claim": "c"}]})
        )
        sub = ai_review.read_submission(path)
        self.assertTrue(sub["submitted"])
        self.assertEqual(len(sub["items"]), 1)
        self.assertEqual(sub["summary"], "s")

    def test_read_submission_coerces_stringified_findings(self) -> None:
        path = self._write(json.dumps({"submitted": True, "findings": "[{\"title\": \"t\"}]"}))
        sub = ai_review.read_submission(path)
        self.assertEqual(len(sub["items"]), 1)

    def test_read_submission_missing_file_is_not_submitted(self) -> None:
        sub = ai_review.read_submission(pathlib.Path("/nonexistent/does-not-exist.json"))
        self.assertFalse(sub["submitted"])
        self.assertEqual(sub["items"], [])

    def test_apply_dedup_clusters_merges_and_escalates(self) -> None:
        cands = {
            "issues": [
                {"issue_id": "AI-001", "severity": "low", "title": "docs drift", "found_by": ["a:m"], "sources": [1]},
                {"issue_id": "AI-002", "severity": "high", "title": "docs out of sync", "found_by": ["b:m"], "sources": [2]},
                {"issue_id": "AI-003", "severity": "medium", "title": "unrelated", "found_by": ["c:m"], "sources": [3]},
            ]
        }
        out = ai_review.apply_dedup_clusters(cands, [["AI-001", "AI-002"]])
        ids = [i["issue_id"] for i in out["issues"]]
        self.assertEqual(ids, ["AI-001", "AI-003"])  # AI-002 merged away
        merged = out["issues"][0]
        self.assertEqual(merged["severity"], "high")  # escalated from low
        self.assertEqual(sorted(merged["found_by"]), ["a:m", "b:m"])

    def test_apply_dedup_clusters_ignores_singletons_and_garbage(self) -> None:
        cands = {"issues": [{"issue_id": "AI-001", "severity": "low", "title": "x", "found_by": [], "sources": []}]}
        # singleton group, unknown id, non-list — all no-ops
        out = ai_review.apply_dedup_clusters(cands, [["AI-001"], ["AI-999", "AI-998"], "junk"])
        self.assertEqual([i["issue_id"] for i in out["issues"]], ["AI-001"])

    def test_parse_name_status_tolerates_malformed_rename(self) -> None:
        # A rename status with a missing field must not IndexError out of the whole review;
        # the well-formed line must still parse.
        rows = ai_review.parse_name_status("R100\tonly_one_field\nM\tfoo.py\n")
        self.assertIn("foo.py", [r["path"] for r in rows])
        # a proper rename still keeps old/new
        rows2 = ai_review.parse_name_status("R100\told.py\tnew.py\n")
        self.assertEqual(rows2[0], {"status": "R", "old_path": "old.py", "path": "new.py"})

    def test_format_location_hides_zero_line(self) -> None:
        self.assertEqual(ai_review.format_location({"file": "a.py", "line": 0}), "a.py")
        self.assertEqual(ai_review.format_location({"file": "a.py", "line": 5}), "a.py:5")

    def test_clean_path_does_not_strip_sibling_prefix(self) -> None:
        old = os.environ.get("GITHUB_WORKSPACE")
        os.environ["GITHUB_WORKSPACE"] = "/ws/repo"
        try:
            self.assertEqual(ai_review.clean_path("/ws/repo/.github/x.py"), ".github/x.py")
            # sibling dir sharing the string prefix must NOT be stripped
            self.assertEqual(ai_review.clean_path("/ws/repo_backup/x.py"), "/ws/repo_backup/x.py".lstrip("/"))
        finally:
            if old is None:
                os.environ.pop("GITHUB_WORKSPACE", None)
            else:
                os.environ["GITHUB_WORKSPACE"] = old

    def test_scoped_provider_env_keeps_only_relevant_key(self) -> None:
        saved = {k: os.environ.get(k) for k in ["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY"]}
        os.environ.update({"OPENROUTER_API_KEY": "or", "ANTHROPIC_API_KEY": "an", "MINIMAX_API_KEY": "mm"})
        try:
            env = ai_review.scoped_provider_env("openrouter/z-ai/glm-5.2")
            self.assertEqual(env.get("OPENROUTER_API_KEY"), "or")
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("MINIMAX_API_KEY", env)
            env2 = ai_review.scoped_provider_env("minimax/MiniMax-M3")
            self.assertEqual(env2.get("MINIMAX_API_KEY"), "mm")
            self.assertNotIn("OPENROUTER_API_KEY", env2)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_clean_path_strips_workspace_prefix(self) -> None:
        old = os.environ.get("GITHUB_WORKSPACE")
        os.environ["GITHUB_WORKSPACE"] = "/home/runner/work/lambda_vm/lambda_vm"
        try:
            self.assertEqual(
                ai_review.clean_path("/home/runner/work/lambda_vm/lambda_vm/.github/scripts/ai_review.py"),
                ".github/scripts/ai_review.py",
            )
            self.assertEqual(
                ai_review.clean_path(".github/scripts/ai_review.py"), ".github/scripts/ai_review.py"
            )
            self.assertEqual(ai_review.clean_path("./docs/ai-review.md"), "docs/ai-review.md")
            self.assertIsNone(ai_review.clean_path("n/a"))
        finally:
            if old is None:
                os.environ.pop("GITHUB_WORKSPACE", None)
            else:
                os.environ["GITHUB_WORKSPACE"] = old

    def test_format_source_cell_breaks_model_onto_own_line(self) -> None:
        cell = ai_review.format_source_cell(["minimax-correctness:minimax/MiniMax-M3"])
        self.assertIn("<br>", cell)
        self.assertEqual(cell, "minimax-correctness<br>minimax/MiniMax-M3")
        self.assertEqual(ai_review.format_source_cell([]), "-")

    def test_format_verifier_label_lists_verifier_lanes(self) -> None:
        label = ai_review.format_verifier_label(
            [{"kind": "verification", "lane_id": "deepseek-verifier", "model": "openrouter/deepseek/deepseek-v4-pro"}]
        )
        self.assertEqual(label, "deepseek-verifier (openrouter/deepseek/deepseek-v4-pro)")

    def test_stream_meta_timeline_records_tool_calls_and_tokens(self) -> None:
        stream = "\n".join(
            [
                json.dumps({"type": "tool_use", "part": {"tool": "read", "state": {"status": "completed", "input": {"filePath": "a.py"}}}}),
                json.dumps({"type": "tool_use", "part": {"tool": "submit_findings", "state": {"status": "completed", "input": {"findings": []}}}}),
                json.dumps({"type": "step_finish", "part": {"tokens": {"output": 0, "reasoning": 6587}}}),
            ]
        )
        meta = ai_review.opencode_stream_meta(stream)
        tools = [e for e in meta["timeline"] if e["t"] == "tool"]
        self.assertEqual([t["tool"] for t in tools], ["read", "submit_findings"])
        steps = [e for e in meta["timeline"] if e["t"] == "step"]
        self.assertEqual(steps[0]["reasoning"], 6587)


if __name__ == "__main__":
    unittest.main()
