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
   (callers, checks, types). A soft budget ends each one at about 50 commands or 30 minutes.
2. A cheap model lists stale docs, missing tests, leftovers and inconsistencies, separately.
3. Duplicate findings are merged. A judge model (GPT 6.1 Sol) checks every finding against the code.
4. One comment is posted. It opens with a table of findings to check: severity, status, which models
   found each one, and where. Collapsed sections hold the details, duplicates, pre-existing bugs,
   dismissed findings with the judge's reason, and housekeeping. The last line shows what the review
   cost.

## What the reviewers focus on

Two files on `main` steer the review; a PR cannot change how it is reviewed:

- [`.github/ai-review/threat_model.md`](../.github/ai-review/threat_model.md): what matters in this
  repository (soundness above all: underconstrained columns, range checks, LogUp/bus balance,
  verifier checks, recursion and continuation linking).
- [`.github/ai-review/config.toml`](../.github/ai-review/config.toml): which paths are
  security-sensitive (they get every reviewer), which are ignored, test file patterns, and the review
  order for large PRs.

## Cost and large PRs

- A PR reviewed in one part costs about $3 (the comment shows each review's cost).
- A PR over about 4,000 changed lines is split into parts of about 2,500 lines. The parts are
  reviewed in priority order (`priority` in the config, then sensitive paths) within a $50 budget.
  Parts left unreviewed are listed at the top of the comment; `/ai-review continue` reviews them.
- The parts of a split PR run one after another, so a very large PR can take hours.

## Secrets

- `AI_REVIEW_OPENROUTER_KEY`: OpenRouter key for every model but Claude (an organization secret,
  shared across repositories). It must come from the OpenRouter account that holds the review's
  provider presets. Its credit limit is the hard ceiling on spend.
- `ANTHROPIC_API_KEY`: Anthropic API key for the Opus reviewer.
