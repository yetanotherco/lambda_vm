# AI Review Workflow

This repository uses manually triggered AI review tiers. Expensive reviewers
should run when the author or reviewer asks for them, not when a draft PR is
opened.

## Commands

Comment on a pull request with one of these commands:

| Command | Tier | Current reviewers | Use when |
| --- | --- | --- | --- |
| `/ai-review standard` | Standard | Kimi | Everyday PRs that are ready for serious review. |
| `/ai-review critical` | Critical | Codex and Claude | Soundness-, security-, VM-, prover-, crypto-, GPU-, or infra-sensitive changes. |
| `/kimi` | Individual | Kimi | Ad-hoc lightweight review. |
| `/codex` | Individual | Codex | Ad-hoc Codex-only review. |
| `/claude` | Individual | Claude | Ad-hoc Claude-only review. |

Only repository owners, members, and collaborators can trigger these reviews.

## Prompt Files

Reviewer prompts live in `.github/ai-review/prompts/` so they can be reused by
any model runner:

- `general.md` backs the individual `/kimi`, `/codex`, and `/claude` commands.
- `standard.md` backs `/ai-review standard`.
- `critical.md` backs `/ai-review critical`.

Model-specific workflows should load one of these prompt files and pass its
contents to the reviewer. Do not duplicate prompt bodies inside model-specific
workflow YAML unless the model adapter requires a small wrapper around the shared
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

Critical review runs Codex and Claude independently. Treat their results as
separate reviewer opinions; a finding should still include concrete evidence in
the changed code.

## Model Matrix Plan

The current implementation reuses existing first-party secrets:

- `OPENAI_API_KEY` for Codex
- `ANTHROPIC_API_KEY` for Claude
- `KIMI_API_KEY` for the current lightweight Kimi lane

When `OPENROUTER_API_KEY` is added, the standard tier should move from the
single Kimi lane to a cheap OpenRouter matrix. Keep Codex and Claude native for
critical review; do not route them through OpenRouter unless there is a clear
reason to give up first-party behavior.

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

Use these rankings as initial guidance only. The review artifacts should track
which model and prompt found each confirmed issue, because local usefulness
matters more than public benchmark rank.

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

The next OpenRouter matrix should write structured artifacts so model quality can
be measured over time:

```text
.ai-review/runs/pr-<number>/
  raw/<lane-id>.json
  candidates.json
  verification.json
  final-issues.json
  model-metrics.json
```

Each final issue should preserve provenance:

```json
{
  "issue_id": "AI-004",
  "status": "confirmed",
  "severity": "high",
  "found_by": ["minimax-m3-bugs", "glm-5.1-reasoning"],
  "verified_by": ["deepseek-v4-pro"],
  "rejected_by": [],
  "file": "prover/src/tables/cpu.rs",
  "line": 123
}
```

Do not count a verifier as `found_by` if it saw candidate findings from another
model. Track discovery and verification separately so we can evaluate:

- confirmed unique discoveries per model and prompt
- false-positive and duplicate rates
- issues found by only one model
- cost and latency per confirmed finding
