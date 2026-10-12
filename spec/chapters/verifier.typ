#import "../meta.typ": todo
// TODO:
// - sigma protocol, layout the various steps
// - Fiat Shamir transformation into non-interactive protocol.
// 

= Auxiliary verification steps
#todo[
  *Verify that the number of tables used by the prover in an epoch/proof is 
  lower than some bound.*
  Without this restriction, a prover can construct an interaction multiplicity 
  of $-1$ by repeating a table row a large number of times.
  While this strategy this would result in a large increase of 
  proof size, the excess can be purged during recursion.
  This could e.g. be used to bypass range checks.

  TODO: determine the right bound. The `IS_WH` chip with its multiplicity of type
  `B20` induces an upper bound: it requires $approx 2^43$ rows 
  (or $approx 2^23$ tables) to achieve a multiplicity sum of $-1$.
  This can then be used to forge an `IS_HALF` check.
]