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

## Reviewer Matrix

API keys are **organization-level** GitHub secrets (not repo-level — `gh secret
list` on the repo won't show them). Each lane's `model` is a provider-qualified
opencode id, so the provider determines which key is used:

- `OPENROUTER_API_KEY` — glm, kimi, nemotron, deepseek lanes, and the minimax-m3
  deduper (everything `openrouter/...`). This key has a **daily spend limit**;
  heavy experimentation can exhaust it (403 "Key limit exceeded (daily limit)").
- `MINIMAX_API_KEY` — the direct `minimax/MiniMax-M3` finder lanes.
- `ANTHROPIC_API_KEY` — the critical-tier native Claude review (opus).
- `OPENAI_API_KEY` — the critical-tier native Codex review.
- `KIMI_API_KEY` → mapped to `MOONSHOT_API_KEY` for the `/kimi` command **only**.
  Kimi in the review swarm goes through **OpenRouter** (`openrouter/moonshotai/...`),
  because the direct Moonshot endpoint rejected the key with `401 Incorrect API
  key`. See "Lessons learned".

A missing key makes only that provider's lanes fail; the report still posts.

### Architecture (agentic, via opencode)

Each lane is **not** a single chat completion. It runs an **opencode** agent in a
read-only sandbox (`.opencode/agent/review-ro.md`) that can `read`/`grep`/`glob`
the repo to explore the change in context, then **reports through a tool call**,
not free-text JSON:

- review lanes call **`submit_findings`** (`.opencode/tools/submit_findings.ts`)
- verifier lanes call **`submit_verifications`** (`.opencode/tools/submit_verifications.ts`)

The tool writes the validated result to `$AI_REVIEW_OUT`, which the orchestrator
reads back. Flow: **finders → heuristic + LLM dedup → verifier → report**. The
matrix (`.github/ai-review/matrix.json`) is per-tier `review_lanes`,
`verifier_lanes`, and a `deduper`. Each lane is `{id, model, prompt, variant}`;
`variant` is opencode's reasoning effort (see "Reasoning effort" below).

All finders use the broad **`general`** prompt (correctness + cosmetic + perf in
one pass), at `low` effort except minimax (`high`, its measured sweet spot — see
"Reasoning effort"). Current **standard** (cheap) matrix:

| Lane | Model | Prompt | Variant |
| --- | --- | --- | --- |
| `glm` | `openrouter/z-ai/glm-5.2` | general | low |
| `kimi` | `openrouter/moonshotai/kimi-k2.7-code` | general | low |
| `nemotron` | `openrouter/nvidia/nemotron-3-ultra-550b-a55b` | general | low |
| `minimax` | `minimax/MiniMax-M3` | general | high |
| `deepseek-verifier` (verify) | `openrouter/deepseek/deepseek-v4-pro` | verify | low |
| deduper | `openrouter/minimax/minimax-m3` | — | low |

Current **critical** (expensive) matrix uses the **same open-weight finder swarm,
`deepseek-v4-pro`/`verify` verifier, and minimax-m3 deduper as standard** — the
structured pipeline is open-weight end-to-end. What makes critical "critical" is
that it *also* triggers the native **Codex** (GPT) and native **Claude** (opus)
reviews, which run in their own vendor harnesses (`critical.md` prompt) and post
their own independent comments. The flagship closed models contribute as
independent native reviews rather than swarm finders: in measured runs the
native Codex pass found a high-severity issue the whole swarm missed, while an
opus *swarm* finder cost ~$1/run for only one unique low finding — so opus was
moved out of the swarm and into its native harness.

Reviewer lanes see the diff plus current/base contents for changed files (size
limited). Verifier lanes see the deduplicated candidates plus the same context.
Final status is `confirmed`, `rejected`, `uncertain`, or `candidate` (no verdict).

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

## Reasoning Effort (`variant`): what we learned

`variant` maps to opencode's provider-specific reasoning effort
(`minimal` < `low` < `medium` < `high` < `max`). It is best-effort: opencode
applies it where the provider supports it and silently ignores it otherwise
(no error), so `low` is a safe default on any lane.

Measured per-model behavior (swept on PR #671 — the AI-review PR itself, ~131KB diff):

| Model | low | high | Takeaway |
| --- | --- | --- | --- |
| minimax-M3 | ~5 | **~43** | high reasons hard over the diff and finds far more (incl. real criticals). Its sweet spot. Also run `max` — it *explores* instead of diff-reasoning and finds **different** issues. |
| glm-5.2 | 3 | 2 | high gives nothing → `low` |
| nemotron-3-ultra | 7 | 0 (explored but never converged) | high is flakier → `low` |
| kimi-k2.7-code | 5 (incl. a critical) | 8 (all medium/low) | at `low` kimi explores files and finds fewer but **higher-severity** issues; at `high` it skips tools, reasons over the diff only, and finds more but **shallower** issues → `low` |

**Key insight: `high` is not universally better.** For most models it makes them
lean on pure diff-reasoning and skip exploration — finding *more but shallower*
issues and missing bugs that require reading files for context. Only **minimax**
clearly benefits from `high`. Everything else is best at `low`, which is cheaper
and less flaky; the verifier and swarm redundancy cover the recall you'd
otherwise chase with `high`. Watch for lanes that explore (many `tool_use`
events) yet submit nothing — that's a reasoning-burn / convergence failure.

## Adding or Changing a Model

1. Add `{id, model, prompt, variant: "low"}` to the tier in
   `.github/ai-review/matrix.json`. Use a provider-qualified opencode id
   (`openrouter/<author>/<model>` or a direct provider id); confirm it exists on
   models.dev and its provider key is in the workflow env.
2. Run the tier on a real PR and read the lane artifact:
   - `submission.submitted == true` with findings → working.
   - `submitted: false` / `event_counts: {step_start: 1}` → emitted nothing
     (reasoning-burn / no convergence). Try another `variant` or drop it.
   - `error` with `401`/`403` → provider auth or OpenRouter daily-cap problem.
   - reads with `status: error` → path/sandbox issue.
3. Tune `variant` UP only if a low-vs-high **sweep** shows real gains for that
   model — don't assume. Sweep by adding `<id>-low` and `<id>-high` lanes and
   comparing findings count **and severity** (count alone misled us on kimi).
   Raise the per-call and wrapper timeouts generously for `high`/`max` lanes.
4. Default new models to `low`; reserve the expensive direct models (Claude/GPT)
   for the critical tier.

## Lessons Learned / Gotchas

- **Report via a tool, not free-text JSON.** Agentic models reliably make tool
  calls but routinely fail the "stop exploring and hand-write the final JSON"
  step (empty output / narration). `submit_findings` / `submit_verifications`
  fixed convergence. Single-shot calls (the deduper) can use free-text JSON
  safely — it's the *agentic loop* that made hand-written JSON fragile.
- **Message on stdin, not argv.** The prompt + diff is piped to opencode on
  stdin; as an argv string it fails with `E2BIG` once the diff crosses ~128KB.
- **Review from the repo root.** opencode's cwd must be the repo root (checkout
  at the workspace root, `--repo .`). With the repo in a `runner/` subdir the
  agent built absolute paths against the workspace root and its reads errored.
  Lane jobs check out at root; other jobs keep their `runner/` checkout.
- **Dedup is two-stage.** A path+text heuristic (`clean_path` normalizes to
  repo-relative via `GITHUB_WORKSPACE`) plus a conservative **LLM dedup** (the
  `deduper`; minimax-m3 won the precision A/B vs deepseek). The LLM call needs a
  generous `max_tokens` (~40k) or reasoning truncates the answer to empty. Dedup
  errs toward under-merging: residual dupes are harmless, over-merging hides a
  finding.
- **`found_by` is provenance.** Both merge stages union it, so the report shows
  every lane (hence variant) that found each issue.
- **OpenRouter vs direct.** OpenRouter mangles tool-calling for some models, so
  agentic lanes prefer direct keys where possible; OpenRouter is fine for cheap
  finders and single-shot calls. Kimi must go via OpenRouter (direct Moonshot
  returned `401`). The OpenRouter key has a **daily spend cap** — heavy
  experimentation exhausts it.
- **Security.** The agent is read-only (`bash`/`edit`/`write`/`patch`/`webfetch`
  denied) with `external_directory: deny`, so it can't read `/proc/self/environ`
  or credential files to leak keys via the report (verified). **Open issue:** the
  workflow installs the agent + tools by copying them *from the PR checkout*, so a
  malicious PR could weaken its own sandbox — install them from the base branch.
- **Diagnostics.** Each lane records an opencode `timeline` (tool calls + args,
  text previews, per-step output/reasoning tokens), `cost`, `tokens`,
  `returncode`, and a stderr tail — that is how every failure above was diagnosed.

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
