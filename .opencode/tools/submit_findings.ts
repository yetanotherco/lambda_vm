import { tool } from "@opencode-ai/plugin"
import { writeFileSync } from "node:fs"

// Structured reporting channel for the review lanes. Instead of asking the model to
// hand-write a JSON blob as its final message (which weak/reasoning models routinely
// fail to do — they explore, then emit empty or narrate), we give it a tool to CALL.
// The validated findings are written to $AI_REVIEW_OUT, which ai_review.py reads back.
export default tool({
  description:
    "Submit your FINAL code-review findings and end the review. Call this EXACTLY ONCE, " +
    "as soon as you have finished reading the relevant code. Report findings ONLY through " +
    "this tool — do not write them as prose. Pass an empty findings array if there are no " +
    "real issues. After calling it, stop: do not call any more tools.",
  args: {
    summary: tool.schema.string().describe("One or two sentence summary of what you reviewed"),
    findings: tool.schema
      .array(
        tool.schema.object({
          severity: tool.schema.enum(["critical", "high", "medium", "low"]),
          confidence: tool.schema.enum(["high", "medium", "low"]),
          title: tool.schema.string().describe("short title"),
          file: tool.schema.string().describe("path/to/file the issue is in"),
          line: tool.schema.number().describe("line number; use 0 if unknown"),
          claim: tool.schema.string().describe("what is wrong"),
          evidence: tool.schema.string().describe("why the code you read supports this"),
          suggested_fix: tool.schema.string().describe("specific fix"),
        }),
      )
      .describe("All findings introduced/exposed by the PR diff; empty array if none"),
  },
  async execute(args) {
    const out = process.env.AI_REVIEW_OUT
    // Models sometimes pass `findings` as a JSON string instead of an array; coerce.
    let findings: unknown = args.findings
    if (typeof findings === "string") {
      try {
        findings = JSON.parse(findings)
      } catch {
        findings = []
      }
    }
    if (!Array.isArray(findings)) findings = []
    const payload = JSON.stringify(
      { submitted: true, summary: args.summary ?? "", findings },
      null,
      2,
    )
    if (out) {
      try {
        writeFileSync(out, payload)
      } catch (e) {
        return `ERROR: could not write findings to ${out}: ${e}. Tell the user this failed.`
      }
    }
    return `Recorded ${(findings as unknown[]).length} finding(s). Review complete — do not call any more tools.`
  },
})
