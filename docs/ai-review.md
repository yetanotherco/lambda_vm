# AI Review

This repository uses the [multi-model review from yetanotherco/actions](https://github.com/yetanotherco/actions#multi-model-pr-review). Reviews run only when a repository owner, member, or collaborator comments `/ai-review` (or `/review-ai`) on a PR, or when someone adds a label starting with `ai-review`. Fork PRs are refused. The review posts one comment with findings checked by a judge, housekeeping notes, and costs.

Put options on the first line of the comment: `/ai-review budget=80` raises this run's budget (up to $100), and `/ai-review continue` resumes an unfinished review of the same commit. The label trigger can test workflow changes before they reach the default branch; comment triggers use the default-branch workflow.

The action reads [config.toml](../.github/ai-review/config.toml) and [threat_model.md](../.github/ai-review/threat_model.md) from the PR base branch. The threat model focuses the reviewers on constraint and verifier soundness. The config excludes Sail-generated Lean translation and other generated or low-value paths from the security reviewers' diff. The action's separate malicious-code check may still read ignored files. A PR's own changes to either file do not affect its review until merged.

The workflow requires the organization secrets `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY`. It pins `yetanotherco/actions` to the `v3.0.0` release commit. See that action's documentation for the reviewer set, size limits, and budget behavior.
