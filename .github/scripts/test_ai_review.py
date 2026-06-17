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

    def test_run_review_lane_keeps_malformed_json_as_parse_warning(self) -> None:
        ai_review.repair_json = None
        raw_response = '''```json
{
  "summary": "tests",
  "findings": [
    {
      "severity": "low",
      "confidence": "high",
      "title": "Missing tests",
      "file": ".github/scripts/ai_review.py",
      "line": 1,
      "claim": "The script has no parser tests.",
      "evidence": "The PR adds parser logic.",
      "suggested_fix": "Add tests for:
1. malformed JSON
2. empty responses"
    }
  ]
}
```'''
        ai_review.openrouter_chat = lambda lane, system, user, api_key: {
            "status": "success",
            "raw_response": raw_response,
            "usage": {},
            "openrouter_id": "test",
        }

        result = ai_review.run_review_lane(self.lane, self.context, "review tests")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["findings"], [])
        self.assertIn("invalid JSON in fenced block", result["parse_error"])

    def test_run_review_lane_treats_empty_response_as_error(self) -> None:
        ai_review.openrouter_chat = lambda lane, system, user, api_key: {
            "status": "success",
            "raw_response": "",
            "usage": {"completion_tokens": 2400},
            "openrouter_id": "test",
        }

        result = ai_review.run_review_lane(self.lane, self.context, "review tests")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "model returned empty response")
        self.assertEqual(result["findings"], [])

    def test_run_review_lane_accepts_valid_findings_wrapper(self) -> None:
        ai_review.openrouter_chat = lambda lane, system, user, api_key: {
            "status": "success",
            "raw_response": """```json
{
  "summary": "one issue",
  "findings": [
    {
      "severity": "low",
      "confidence": "high",
      "title": "Missing tests",
      "file": ".github/scripts/ai_review.py",
      "line": 1,
      "claim": "Parser behavior is untested.",
      "evidence": "The changed script handles malformed model output.",
      "suggested_fix": "Add parser tests."
    }
  ]
}
```""",
            "usage": {},
            "openrouter_id": "test",
        }

        result = ai_review.run_review_lane(self.lane, self.context, "review tests")

        self.assertEqual(result["status"], "success")
        self.assertNotIn("parse_error", result)
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings"][0]["title"], "Missing tests")


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

    def test_run_verifier_lane_normalizes_verifications(self) -> None:
        ai_review.openrouter_chat = lambda lane, system, user, api_key: {
            "status": "success",
            "raw_response": """```json
{
  "summary": "checked",
  "verifications": [
    {
      "issue_id": "AI-001",
      "status": "confirmed",
      "confidence": "high",
      "rationale": "The parser behavior follows from the diff."
    },
    {
      "issue_id": "AI-002",
      "status": "not-sure",
      "confidence": "low",
      "rationale": "Invalid status should normalize to uncertain."
    }
  ]
}
```""",
            "usage": {},
            "openrouter_id": "test",
        }

        result = ai_review.run_verifier_lane(self.lane, self.context, self.candidates, "verify")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["summary"], "checked")
        self.assertEqual([item["status"] for item in result["verifications"]], ["confirmed", "uncertain"])
        self.assertEqual(result["verifications"][0]["verifier"], "qwen-standard-verifier:qwen/qwen3.7-plus")

    def test_run_verifier_lane_treats_empty_response_as_error(self) -> None:
        ai_review.openrouter_chat = lambda lane, system, user, api_key: {
            "status": "success",
            "raw_response": "",
            "usage": {"completion_tokens": 2600},
            "openrouter_id": "test",
        }

        result = ai_review.run_verifier_lane(self.lane, self.context, self.candidates, "verify")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "model returned empty response")
        self.assertEqual(result["verifications"], [])

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
