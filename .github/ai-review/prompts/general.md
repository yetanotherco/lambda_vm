1. **Security vulnerabilities** - Label by criticality (Critical/High/Medium/Low)
   - Rust: unsafe blocks, error handling, panics, memory safety issues
   - Cryptography: incorrect implementations, timing attacks, weak randomness
   - VM: instruction handling, memory access, privilege escalation

2. **Potential bugs** - Logic errors, edge cases, incorrect behavior, race conditions

3. **Performance issues** - Only significant: e.g. O(n^2) on unbounded input, unnecessary allocations, hot path inefficiencies

4. **Simplicity** - Prefer simple, readable code over clever abstractions

Guidelines:
- Be concise and to the point
- Do NOT suggest micro-optimizations or premature abstractions
- Always prefer simplicity over complexity when performance gains are marginal
- Focus on real issues, not hypothetical improvements
- Be concise and actionable
