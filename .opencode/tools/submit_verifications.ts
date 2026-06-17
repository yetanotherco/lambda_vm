import { tool } from "@opencode-ai/plugin"
import { writeFileSync } from "node:fs"

// Structured reporting channel for verifier lanes — the mirror of submit_findings.
// The verifier confirms/rejects each candidate finding and reports the verdicts by
// CALLING this tool (reliable) rather than hand-writing a final JSON blob (unreliable).
export default tool({
  description:
    "Submit your FINAL verification verdicts and end the task. Call this EXACTLY ONCE, " +
    "after you have checked each candidate issue against the code. Provide one entry per " +
    "issue_id you were asked to verify. Report ONLY through this tool — do not write the " +
    "verdicts as prose. After calling it, stop: do not call any more tools.",
  args: {
    summary: tool.schema.string().describe("One or two sentence summary of the verification"),
    verifications: tool.schema
      .array(
        tool.schema.object({
          issue_id: tool.schema.string().describe("the AI-### id of the candidate issue"),
          status: tool.schema.enum(["confirmed", "rejected", "uncertain"]),
          confidence: tool.schema.enum(["high", "medium", "low"]),
          rationale: tool.schema.string().describe("why, grounded in the code you read"),
        }),
      )
      .describe("One verdict per candidate issue_id"),
  },
  async execute(args) {
    const out = process.env.AI_REVIEW_OUT
    let verifications: unknown = args.verifications
    if (typeof verifications === "string") {
      try {
        verifications = JSON.parse(verifications)
      } catch {
        verifications = []
      }
    }
    if (!Array.isArray(verifications)) verifications = []
    const payload = JSON.stringify(
      { submitted: true, summary: args.summary ?? "", verifications },
      null,
      2,
    )
    if (out) {
      try {
        writeFileSync(out, payload)
      } catch (e) {
        return `ERROR: could not write verifications to ${out}: ${e}. Tell the user this failed.`
      }
    }
    return `Recorded ${(verifications as unknown[]).length} verdict(s). Done — do not call any more tools.`
  },
})
