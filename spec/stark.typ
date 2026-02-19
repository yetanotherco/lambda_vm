#import "/book.typ": book-page

#show: book-page("stark.typ")

#set heading(numbering: "I.1.a.")

= Notation

== General notation

#let BaseField = math.FF
#let ExtensionField = math.GG

=== Sets

- $NN$: the set of non-zero natural integers.
- $[n]$ for $n in NN$: the set of integers ${0, dots, n - 1}$.
- $BaseField$: the base finite field used by the arithmetisation.
- $ExtensionField$: a finite extension of $BaseField$ of cryptographic size.

=== Tuples

- $X[i]$ for tuple $X$: the $i$-th element of $X$, starting at $0$.

== Arithmetisation notation

=== Tables

#let numTables = $sans(t)$
#let Table = $T$
#let TableSet = ${Table_i}_(i in [t])$
#let numColumns = $sans(m)$
#let numRows = $sans(n)$

- $numTables in NN$: number of tables $Table_i$ in the arithmetisation of the VM.
- $Table_i in BaseField^(numRows_i times numColumns_i)$: a table in the arithmetisation of the VM; $TableSet$ denotes set of all tables.
- $numColumns_i in NN$: number of _columns_ in table $Table_i$ (not the number of variables).
- $numRows_i in NN$: number of _rows_ in table $Table_i$.

=== Constraints

#let indX = $cal(X)$
#let indY = $cal(Y)$
#let indTuple = $arrow(indX)$
#let numConstraints = $sans(c)$
#let ConstraintSet = $scr(C)$
#let Constraint = $cal(C)$
#let EnforcementDom = $scr(H)$

- $indTuple_i = (indX_(i,1), dots, indX_(i, numColumns_i), indY_(i,1), dots, indY_(i, numColumns_i))$: _tuple of $2 numColumns_i$ indeterminates_, two for each column of a table: one $indX$ for the current row, and one $indY$ for the following row.
- $numConstraints_i in NN$: _number of constraints_ for table $Table_i$.
- $ConstraintSet_i subset BaseField[indTuple_i]$: _set of constraints_ for table $Table_i$, of size $numConstraints_i$.
- $Constraint_(i,j) in BaseField[indTuple_i]$: $j$-th _constraint polynomial_ for $Table_i$, for $j in [numConstraints_i]$.
- $EnforcementDom_(i,j) subset [numRows_i]$: _enforcement domain_ for the $j$-th constraint of table $Table_i$, selecting the rows for which the $j$-th constraint equation is expecting to hold.
- $j$-th _constraint equation_ for table $Table_i$: $ forall k in EnforcementDom_(i,j), quad Constraint_(i,j)(Table_i [k] || Table_i [k+1]) = 0_BaseField. $
