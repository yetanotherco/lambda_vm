# AI Review Workflow

This repository uses a single, manually triggered AI review flow. It is
deliberately opt-in: expensive reviewers run when the author or a reviewer asks
for them, never automatically on PR open.

## Commands

Comment `/ai-review` on a pull request to run the review. There is one flow —
no standard/critical distinction. (A trailing word like `/ai-review critical` is
tolerated and runs the same thing, but isn't needed.)

| Command | Reviewers | Use when |
| --- | --- | --- |
| `/ai-review` | Open-weight swarm + verifier (structured report), plus native Codex and Claude (opus) | Any PR worth a serious review — especially soundness-, security-, VM-, prover-, crypto-, GPU-, or infra-sensitive changes. |

You can also add the `ai-review` label to a pull request. (The older
`ai-review-standard` / `ai-review-critical` labels still trigger the same flow,
kept for back-compat.) The label trigger is useful for testing workflow changes
before they are merged, because `pull_request` label events run against the PR
workflow definition.

> **Note:** the **native Claude** review and the `/ai-review` **comment** trigger
> only activate once this workflow is merged to the default branch.
> `claude-code-action` refuses to run unless the invoking workflow is identical to
> the version on `main` (an anti-pwn-request guard), and `issue_comment` always
> uses the default-branch workflow. Pre-merge, use the **label** trigger: the
> swarm and native Codex run, but native Claude self-skips until merge.

Comment commands are restricted to repository owners, members, and
collaborators. Label triggers are controlled by GitHub's label permissions.

## Prompt Files

Reviewer prompts live in `.github/ai-review/prompts/` so they can be reused by
any model runner:

- `general.md` is the review prompt used by every swarm lane **and** by the
  native Codex/Claude reviews (passed as their `custom_prompt` input). There is
  one generic review prompt; there is intentionally no separate soundness brief
  (see "Lessons learned").
- `lanes/verify.md` is the verifier prompt.

Model-specific workflows should load one of these prompt files and pass its
contents to the reviewer. Do not duplicate prompt bodies inside model-specific
workflow YAML unless the model adapter requires a small wrapper around the shared
prompt.

The model-to-prompt mapping lives in `.github/ai-review/matrix.json`. Prompts
are intentionally model-agnostic; the matrix decides which model receives which
prompt.

## What the review covers

The review is one flow with two independent parts, and **both use the same
generic `general.md` prompt**. It focuses on:

- correctness and regressions introduced by the branch
- safety/security: unsafe Rust, panics, memory safety, resource exhaustion
- local constraint, trace, and bus consistency when those files change
- VM/executor behavior, memory access, state transitions
- missing tests or changed test intent
- simplicity, maintainability, stale comments/names/docs, scope drift

**1. Structured swarm** (open-weight finders + verifier) → one deduplicated
report with per-finding provenance.

**2. Native Codex + Claude (opus) reviews** run independently in the vendors'
own harnesses and post their own comments. Treat them as separate reviewer
opinions; they are not included in the structured provenance report. They run
flagship models in full agentic harnesses, so they tend to explore deeper than
the constrained swarm — but they get the **same generic prompt**, not a
soundness brief.

**Soundness is a deliberate gap.** Neither part is equipped to find real
soundness bugs (under-constrained AIRs, transcript/Fiat-Shamir/commitment
mistakes, witness-soundness drift). A generic prompt that merely *names* those
topics does not help a model find them — soundness review needs dedicated
tooling (concrete failure patterns, spec context, targeted reasoning) and is
deferred to that future work, not attempted here.

## Reviewer Matrix

API keys are **organization-level** GitHub secrets (not repo-level — `gh secret
list` on the repo won't show them). Each lane's `model` is a provider-qualified
opencode id, so the provider determines which key is used:

- `OPENROUTER_API_KEY` — glm, kimi, nemotron, deepseek lanes, and the minimax-m3
  deduper (everything `openrouter/...`). This key has a **daily spend limit**;
  heavy experimentation can exhaust it (403 "Key limit exceeded (daily limit)").
- `MINIMAX_API_KEY` — the direct `minimax/MiniMax-M3` finder lanes.
- `ANTHROPIC_API_KEY` — the native Claude review (opus).
- `OPENAI_API_KEY` — the native Codex review.
- `KIMI_API_KEY` (→ `MOONSHOT_API_KEY`) is **no longer used** — the standalone
  `/kimi` command was retired. Kimi in the review swarm goes through **OpenRouter**
  (`openrouter/moonshotai/...`), because the direct Moonshot endpoint rejected the
  key with `401 Incorrect API key`. See "Lessons learned".

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
matrix (`.github/ai-review/matrix.json`) holds the single flow's `review_lanes`,
`verifier_lanes`, and a `deduper` (flat — there is no tier key).
Each lane is `{id, model, prompt, variant}`; `variant` is opencode's reasoning
effort (see "Reasoning effort" below).

All finders use the broad **`general`** prompt (correctness + cosmetic + perf in
one pass), at `low` effort except minimax (`high`, its measured sweet spot — see
"Reasoning effort"). The structured swarm is **open-weight end-to-end**:

| Lane | Model | Prompt | Variant |
| --- | --- | --- | --- |
| `glm` | `openrouter/z-ai/glm-5.2` | general | low |
| `kimi` | `openrouter/moonshotai/kimi-k2.7-code` | general | low |
| `nemotron` | `openrouter/nvidia/nemotron-3-ultra-550b-a55b` | general | low |
| `minimax` | `minimax/MiniMax-M3` | general | high |
| `deepseek-verifier` (verify) | `openrouter/deepseek/deepseek-v4-pro` | verify | low |
| deduper | `openrouter/minimax/minimax-m3` | — | low |

Alongside the swarm, the flow **also** triggers the native **Codex** (GPT) and
native **Claude** (opus) reviews — they run in their own vendor harnesses (with
the same generic `general.md` prompt) and post their own independent comments,
outside the structured report. The flagship closed models contribute as independent native
reviews rather than swarm finders: in measured runs the native Codex pass found
a high-severity issue the whole swarm missed, while an opus *swarm* finder cost
~$1/run for only one unique low finding — so opus was moved out of the swarm and
into its native harness.

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

1. Add `{id, model, prompt, variant: "low"}` to `review_lanes` (or
   `verifier_lanes`) in `.github/ai-review/matrix.json`. Use a provider-qualified
   opencode id (`openrouter/<author>/<model>` or a direct provider id); confirm it
   exists on models.dev and its provider key is in the workflow env.
2. Run the review on a real PR and read the lane artifact:
   - `submission.submitted == true` with findings → working.
   - `submitted: false` / `event_counts: {step_start: 1}` → emitted nothing
     (reasoning-burn / no convergence). Try another `variant` or drop it.
   - `error` with `401`/`403` → provider auth or OpenRouter daily-cap problem.
   - reads with `status: error` → path/sandbox issue.
3. Tune `variant` UP only if a low-vs-high **sweep** shows real gains for that
   model — don't assume. Sweep by adding `<id>-low` and `<id>-high` lanes and
   comparing findings count **and severity** (count alone misled us on kimi).
   Raise the per-call and wrapper timeouts generously for `high`/`max` lanes.
4. Default new models to `low`. Keep the expensive flagship closed models
   (Claude/GPT) out of the swarm — they contribute via the native Codex/Claude
   reviews instead.

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
- **No soundness prompt (yet).** The swarm and the native reviews share one
  generic `general.md`. A prompt that merely *names* soundness topics
  (Fiat-Shamir, commitments, AIR inclusion, witness-soundness) does not help a
  model find soundness bugs — those need counterexample reasoning, spec
  knowledge, and knowing what a constraint must enforce. Naming the topics just
  *looks* like coverage we lack. Real soundness review is deferred to dedicated
  tooling; the generic prompt honestly targets correctness/security, not
  soundness.
- **OpenRouter vs direct.** OpenRouter mangles tool-calling for some models, so
  agentic lanes prefer direct keys where possible; OpenRouter is fine for cheap
  finders and single-shot calls. Kimi must go via OpenRouter (direct Moonshot
  returned `401`). The OpenRouter key has a **daily spend cap** — heavy
  experimentation exhausts it.
- **Security — the agent sandbox is not the main control.** The agent is
  read-only (`bash`/`edit`/`write`/`patch`/`webfetch` denied) with
  `external_directory: deny`, so the *LLM* can't read `/proc/self/environ` to
  leak keys (verified). But the sandbox does **not** stop PR-controlled *code*
  (`ai_review.py`, `.opencode/tools/*.ts`) from exfiltrating: that code runs as
  the workflow step, with the provider secrets in its env. This is a "pwn
  request": the danger is *whose code runs*, not who triggers — a trusted member
  running `/ai-review` on an external PR would execute that PR's code with the
  secrets.
- **Mitigation: refuse fork PRs — in the trusted layer.** Only same-repo
  branches (which require write access) may reach the secret-bearing,
  code-executing steps. This must be enforced in *trusted* code: on the
  `pull_request` (label) arm `prepare` runs `ai_review.py` checked out **from the
  PR**, so a fork could rewrite the gate itself — that arm is therefore gated in
  the **workflow `if`** using the trusted event context
  (`head.repo.full_name == base.repo.full_name`), before any checkout, so a fork
  PR's job never starts. The `issue_comment` arm runs `prepare` from the default
  branch (trusted), so its fork gate is the `pr_is_from_fork` check there (the
  comment event lacks head-repo info for the `if`); that check is also
  defense-in-depth everywhere. (`pull_request` additionally withholds secrets and
  the write token from forks by default.) Comment triggers are gated to
  OWNER/MEMBER/COLLABORATOR. Lane ids are validated to `[A-Za-z0-9._-]` and passed
  via env (not raw `${{ }}` shell interpolation) to close matrix→shell injection.
  The same trusted same-repo `if` is also replicated on every downstream job that
  holds secrets or the write token (`openrouter-review`, `candidates`,
  `openrouter-verify`, `final-report`) so the gate isn't a single transitive
  choke point. Model-supplied finding text is HTML-escaped before it goes into the
  posted comment, and the `submit_*` tools only write to the orchestrator's
  expected `lane-*.submit.json` path. The lane jobs run under harden-runner
  `egress-policy: block` with an allowlist (GitHub infra, opencode install/binary/
  catalog, pip + npm, and the model APIs `openrouter.ai` / `api.minimax.io`), and
  the opencode installer script is fetched with a pinned sha256 — so a compromised
  dependency or installer can't exfiltrate to an arbitrary host. The allowlist was
  harvested from a real run's audit; adding a new direct provider means adding its
  host to `allowed-endpoints` or that lane is blocked.
  Residual (accepted): a *write-access* user could still run malicious code with
  the secrets — they can already reach secrets via other workflows, so it's
  within the trust boundary. The fuller fix (run trusted runner code from the
  base ref, check out the PR only as read-only review data) is a future option;
  it has a bootstrapping circularity and the same effective boundary.
- **Diagnostics.** Each lane records an opencode `timeline` (tool calls + args,
  text previews, per-step output/reasoning tokens), `cost`, `tokens`,
  `returncode`, and a stderr tail — that is how every failure above was diagnosed.

## One prompt for all reviewers

The system uses a single generic prompt (`general.md`) for every reviewer — the
open-weight swarm finders and the native Codex/Claude reviews alike. An earlier
design used multiple focused prompts per model; it was dropped because the
structured swarm converges better on one broad prompt and a per-model prompt
matrix wasn't worth the upkeep. There is intentionally no separate soundness
prompt — see "Lessons learned" for why.

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
ai-review-final-<pr-number>/
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
  "found_by": ["nemotron:openrouter/nvidia/nemotron-3-ultra-550b-a55b", "glm:openrouter/z-ai/glm-5.2"],
  "verified_by": ["deepseek-verifier:openrouter/deepseek/deepseek-v4-pro"],
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
