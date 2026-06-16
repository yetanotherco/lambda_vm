# AI Review Workflow

This repository uses manually triggered AI review tiers. Expensive reviewers
should run when the author or reviewer asks for them, not when a draft PR is
opened.

## Commands

Comment on a pull request with one of these commands:

| Command | Tier | Current reviewers | Use when |
| --- | --- | --- | --- |
| `/ai-review standard` | Standard | OpenRouter matrix + verifier | Everyday PRs that are ready for serious review. |
| `/ai-review critical` | Critical | OpenRouter matrix + verifiers, plus native Codex and Claude | Soundness-, security-, VM-, prover-, crypto-, GPU-, or infra-sensitive changes. |
| `/kimi` | Individual | Kimi | Ad-hoc lightweight review. |
| `/codex` | Individual | Codex | Ad-hoc Codex-only review. |
| `/claude` | Individual | Claude | Ad-hoc Claude-only review. |

You can also add one of these labels to a pull request:

| Label | Tier |
| --- | --- |
| `ai-review-standard` | Standard |
| `ai-review-critical` | Critical |

The label trigger is useful for testing workflow changes before they are merged,
because `pull_request` label events run against the PR workflow definition.

Comment commands are restricted to repository owners, members, and
collaborators. Label triggers are controlled by GitHub's label permissions.

## Prompt Files

Reviewer prompts live in `.github/ai-review/prompts/` so they can be reused by
any model runner:

- `general.md` backs the individual `/kimi`, `/codex`, and `/claude` commands.
- `standard.md` backs `/ai-review standard`.
- `critical.md` backs `/ai-review critical`.
- `lanes/*.md` backs focused OpenRouter review and verification lanes.

Model-specific workflows should load one of these prompt files and pass its
contents to the reviewer. Do not duplicate prompt bodies inside model-specific
workflow YAML unless the model adapter requires a small wrapper around the shared
prompt.

The model-to-prompt mapping lives in `.github/ai-review/matrix.json`. Prompts
are intentionally model-agnostic; the matrix decides which model receives which
prompt.

## Tier Policy

### Standard

Use standard review for most PRs after they are ready for review. The goal is a
serious, high-signal review using the standard-cost reviewer set, not a final
certification.

The standard reviewer focuses on:

- correctness and regressions introduced by the branch
- local constraint, trace, and bus consistency when those files change
- missing tests or changed test intent
- simplicity and maintainability
- stale comments, stale names, misleading docs, and scope drift

Standard review is allowed to review constraint changes in the PR. It is not a
proof-system or transcript design audit.

### Critical

Use critical review when a small change can still have high impact. Size is not
the deciding factor. Trigger critical review for changes touching:

- soundness-sensitive prover constraints, trace generation, buses, AIR
  inclusion, or statements
- VM, executor, memory, CPU, ALU, load/store, branch, decode, or halt behavior
- hashing, Fiat-Shamir transcripts, FRI, Merkle commitments, challenge
  derivation, or broader prover/verifier soundness assumptions
- GPU/CUDA proving paths
- security-sensitive infra or CI behavior
- merge-conflict resolutions in high-risk branches

Critical review also triggers native Codex and Claude independently. Treat their
results as separate reviewer opinions; they currently post their own comments
and are not included in the structured OpenRouter provenance report.

## OpenRouter Matrix

`/ai-review standard` and `/ai-review critical` require `OPENROUTER_API_KEY`.
If the secret is missing, the workflow still posts a report, but the OpenRouter
lanes are marked as skipped.

The current implementation uses these secrets:

- `OPENROUTER_API_KEY` for the structured matrix, verification, artifacts, and
  final report
- `OPENAI_API_KEY` for Codex
- `ANTHROPIC_API_KEY` for Claude
- `KIMI_API_KEY` for the individual `/kimi` command

Standard review lanes:

| Lane | Model | Prompt |
| --- | --- | --- |
| `minimax-correctness` | `minimax/minimax-m3` | `correctness` |
| `minimax-maintainability` | `minimax/minimax-m3` | `maintainability` |
| `mimo-tests` | `xiaomi/mimo-v2.5` | `tests` |
| `glm-standard` | `z-ai/glm-5.1` | `standard` |
| `qwen-standard-verifier` | `qwen/qwen3.7-plus` | `verify` |

Critical review lanes:

| Lane | Model | Prompt |
| --- | --- | --- |
| `minimax-critical-correctness` | `minimax/minimax-m3` | `correctness` |
| `minimax-critical-maintainability` | `minimax/minimax-m3` | `maintainability` |
| `deepseek-soundness` | `deepseek/deepseek-v4-pro` | `soundness` |
| `glm-critical` | `z-ai/glm-5.1` | `critical` |
| `qwen-critical` | `qwen/qwen3.7-max` | `critical` |
| `glm-critical-verifier` | `z-ai/glm-5.1` | `verify-critical` |
| `deepseek-critical-verifier` | `deepseek/deepseek-v4-pro` | `verify-critical` |

Reviewer lanes see the diff plus current and base contents for changed files,
within size limits. Verifier lanes see the deduplicated candidate findings plus
the same PR context. Verification status is `confirmed`, `rejected`,
`uncertain`, or `candidate` when no verifier result is available.

OpenRouter catalog snapshot from 2026-06-16:

| Model | Input $/1M | Output $/1M | Context | Coding index | Agentic index | Design code rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek/deepseek-v4-flash` | 0.098 | 0.196 | 1,048,576 | 38.7 | 61.3 | 27 |
| `xiaomi/mimo-v2.5` | 0.14 | 0.28 | 1,048,576 | 42.1 | 65.5 | 12 |
| `minimax/minimax-m3` | 0.30 | 1.20 | 1,048,576 | 43.4 | 68.6 | 11 |
| `qwen/qwen3.7-plus` | 0.32 | 1.28 | 1,000,000 | 46.5 | 65.1 | n/a |
| `deepseek/deepseek-v4-pro` | 0.435 | 0.87 | 1,048,576 | 47.5 | 67.2 | 16 |
| `xiaomi/mimo-v2.5-pro` | 0.435 | 0.87 | 1,048,576 | 45.5 | 67.4 | 8 |
| `moonshotai/kimi-k2.7-code` | 0.75 | 3.50 | 262,144 | n/a | n/a | 9 |
| `z-ai/glm-5.1` | 0.98 | 3.08 | 202,752 | 43.4 | 67.1 | 4 |
| `qwen/qwen3.7-max` | 1.25 | 3.75 | 1,000,000 | 50.1 | 66.6 | 10 |

Use these rankings as initial guidance only. The review artifacts track which
model and prompt found each issue, because local usefulness matters more than
public benchmark rank.

## Multiple Prompts Versus One Prompt

Use multiple prompts when both conditions hold:

- the model is cheap enough that repeated input is acceptable
- the model benefits from a narrow lens and may blur tasks in a broad prompt

Use one broad prompt when either condition holds:

- the model is expensive enough that repeated full-context input dominates cost
- the model handles multi-objective review well enough in one pass

Initial policy:

| Model family | Prompt strategy | Reason |
| --- | --- | --- |
| MiMo V2.5 | Multiple focused prompts | Extremely cheap; use for stale comments, missing tests, edge cases, and adversarial sanity checks. |
| MiniMax M3 | Multiple focused prompts | Cheap enough for repeated passes and strong enough to be a workhorse. |
| DeepSeek V4 Flash | One or two focused prompts | Very cheap; good for adversarial or regression-focused checks. |
| Qwen 3.7 Plus | One broad prompt | Strong cheap generalist; avoid redundant repeated input until local data says otherwise. |
| Kimi K2.7 Code | One code-focused prompt | More expensive output and smaller context; use as a coding specialist. |
| GLM 5.1 | One reasoning-focused prompt | More expensive; use for broad correctness reasoning, not repeated cheap lanes. |
| Codex / GPT-5.5 | One broad pass or targeted verification | Expensive; reserve repeated use for critical findings. |
| Claude Sonnet/Opus/Fable | One broad pass or targeted disagreement review | Expensive; use for critical PRs or to challenge Codex findings. |

## Evaluation Artifacts

The OpenRouter workflow writes structured artifacts so model quality can be
measured over time:

```text
ai-review-context-<pr-number>/
  context.json
  pr.diff
ai-review-lane-<lane-id>/
  <lane-id>.json
ai-review-candidates-<pr-number>/
  candidates.json
  model-metrics.json
ai-review-verification-<lane-id>/
  <lane-id>.json
ai-review-final-<tier>-<pr-number>/
  final-issues.json
  model-metrics.json
  report.md
```

Each final issue should preserve provenance:

```json
{
  "issue_id": "AI-004",
  "status": "confirmed",
  "severity": "high",
  "found_by": ["minimax-correctness:minimax/minimax-m3", "glm-standard:z-ai/glm-5.1"],
  "verified_by": ["qwen-standard-verifier:qwen/qwen3.7-plus"],
  "rejected_by": [],
  "file": "prover/src/tables/cpu.rs",
  "line": 123
}
```

Do not count a verifier as `found_by` if it saw candidate findings from another
model. Discovery and verification are tracked separately so we can evaluate:

- confirmed unique discoveries per model and prompt
- false-positive and duplicate rates
- issues found by only one model
- cost and latency per confirmed finding
