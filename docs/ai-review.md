# AI Review Workflow

This repository uses the multi-model AI security review from
[`yetanotherco/actions`](https://github.com/yetanotherco/actions#multi-model-pr-review)
(`pr_review_multi.yml`). It is opt-in: it runs when someone asks for it, never automatically on PR
open.

## Commands

Comment on a pull request (the first line of the comment):

| Comment | Effect |
| --- | --- |
| `/ai-review` (or `/review-ai`) | Review the PR |
| `/ai-review budget=80` | Use $80 for this review instead of the default (at most $100) |
| `/ai-review continue` | Review only the parts an earlier, budget-limited review of the same commit skipped |

Adding a label that starts with `ai-review` also runs the review. Comment commands always run the
workflow from `main`; `pull_request` label events run the PR's own workflow, so use the label to test
changes to the workflow before they are merged.

Only repository owners, members and collaborators can trigger a review, and fork PRs are not reviewed.

## What it does

1. Five models review the PR in parallel: Opus 5.5, GPT 6.1 Sol, Kimi K3.1, GLM 5.3 and DeepSeek V4
   Pro. Each one is an agent in a read-only, offline sandbox, so it reads code outside the diff
   (callers, checks, types). A soft budget asks each one to finish after 80 commands (120 for GPT and
   GLM), about $1.50–2.50 of estimated spend, or 30 minutes (the command and spend limits double
   for priority parts of a split PR).
2. A cheap model lists stale docs, missing tests, leftovers and inconsistencies, separately.
3. Duplicate findings are merged. A judge model (GPT 6.1 Sol) checks the findings against the code;
   the comment says how many it checked, and marks the rest *unverified*.
4. One comment is posted. It opens with a table of findings to check: severity, status, which models
   found each one, and where. Collapsed sections hold the details, duplicates, pre-existing bugs,
   dismissed findings with the judge's reason, and housekeeping. The last line shows what the review
   cost.

## What the reviewers focus on

Two files steer the review. They are read from the PR's base branch, so a PR's own changes to them
do not apply to its review (the label trigger does run the PR's version of the workflow file):

- [`.github/ai-review/threat_model.md`](../.github/ai-review/threat_model.md): what matters in this
  repository (soundness above all: underconstrained columns, range checks, LogUp/bus balance,
  verifier checks, recursion and continuation linking).
- [`.github/ai-review/config.toml`](../.github/ai-review/config.toml): which paths are ignored, test
  file patterns, and the review order for large PRs (`priority`; those parts also get twice the
  per-reviewer budget). Every reviewed file counts as security-sensitive, so every part gets all five
  reviewers.

## Cost and large PRs

- A PR reviewed in one part costs about $3 (the comment shows each review's cost).
- A PR over about 4,000 *effective* lines (changed code, whitespace-only changes and long deletions
  compacted, plus half the test lines) is split into parts of about 2,500 lines. The parts are
  reviewed in priority order (`priority` in the config, then the paths listed as `sensitive`) within
  a $50 budget. Parts left unreviewed are listed at the top of the comment; `/ai-review continue`
  reviews them.
- The parts of a split PR run one after another. No new part starts after 240 minutes, so a very
  large PR ends with a partial review and `continue` does the rest.

## Secrets

- `AI_REVIEW_OPENROUTER_KEY`: OpenRouter key for every model but Claude (an organization secret,
  shared across repositories). It must come from the OpenRouter account that holds the review's
  provider presets. Its credit limit is the hard ceiling on those models' spend.
- `ANTHROPIC_API_KEY`: Anthropic API key for the Opus reviewer. Opus is billed by Anthropic, outside
  the OpenRouter limit: give its workspace a monthly spend limit.

## Changes from the previous review

The opencode swarm (`.github/scripts/ai_review.py`, its prompts and `.opencode/`) is gone.

- Its separate native Claude and Codex comments are gone too: Opus and GPT are among the five
  reviewers of the single comment.
- The `harden-runner` egress allowlist is gone; each reviewer runs in a sandbox with no network
  instead.
- The comment must start with the command (it used to match anywhere in the comment).
- A second request waits for the running review instead of cancelling it.
- The secrets `OPENAI_API_KEY`, `MINIMAX_API_KEY`, `ZRO_API_KEY` and `OPENROUTER_API_KEY` are no
  longer used by this workflow.
