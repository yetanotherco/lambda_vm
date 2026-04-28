#import "/book.typ": book-page, aside, cdsg

#show: book-page("logup")
#show link: underline

#show "constraint choice": link(<constraint_choices>)[constraint choice]

The _LogUp_ proof system conducts a permutation check based on summing partial derivatives. This check ensures that whatever tuple is sent to be "looked-up" by a _source table_ is indeed received in the expected _destination table_.

= Notation

#let BaseField = math.FF
#let ExtensionField = math.GG

== VM Notation

=== Preliminary notation
- $NN$: the set of non-negative natural integers.
- $BaseField$: the base finite field used by the arithmetisation.
- $ExtensionField$: a finite extension of $BaseField$ of cryptographic size.
- $[n]$ for $n in NN$: the set of integers ${0, dots, n - 1}$.
- $X[i]$ for tuple $X$: the $i$-th element of $X$, starting at $0$.

=== Arithmetisation notation

#let numTables = $sans(t)$
#let Table = $T$
#let TableSet = ${Table_i}_(i in [t])$
#let numColumns = $sans(m)$
#let numRows = $sans(N)$

- $numTables in NN$: number of tables $Table_i$ in the arithmetisation of the VM.
- $TableSet$: set of all tables $Table_i$ in the arithmetisation of the VM.
- $numColumns_i in NN$: number of _columns_ in table $Table_i$ (not the number of variables).
- $numRows_i in NN$: number of _rows_ in table $Table_i$.

== Interaction Notation

#let Interaction = $I$
#let id = $sans(id)$
#let numElements = $ell$
#let weightFunction = $w$
#let multiplicity = $mu$

The $j$-th _interaction_ $Interaction_j$ of table $Table_i$ is defined by the following tuple:

#table(
  columns: (auto, auto),
  inset: 6pt,
  align: horizon,
  stroke: none,
  table.header([*Symbol*], [*Description*]),
  table.hline(stroke: 1pt),
  table.vline(stroke: 1pt, x: 1),
  [$id_(i,j) in FF$], 
  [the _type identifier_ of the interaction, usually the identifier of the chip that is constraining the relation expected to hold within the looked-up tuple.],
  [$numElements_(i,j) in NN$], 
  [the _length_ of the tuple of elements being looked-up.],
  [
    $weightFunction_(i,j) : FF^(numColumns_i) & arrow FF^(numElements_(i,j) + 1) \
    R & mapsto arrow(t)_(i,j) || mu_(i,j)$
  ],
  [the _weight function_ that maps a row $R$ of table $Table_i$ to the looked-up tuple $arrow(t)_(i,j)$ and its multiplicity $mu_(i,j) in BaseField$.],
)


= Vanilla LogUp

== Protocol Description

#let logupChallenge = math.alpha
#let fingerprintCoeff = math.beta

#set enum(numbering: "1.a.i.1.a.")

+ Prover commits to all traces.

+ Verifier samples a random _(global) LogUp challenge_ $logupChallenge in ExtensionField$ and a random _fingerprint coefficient_ $fingerprintCoeff in ExtensionField$ and sends them to the Prover.

+ Prover commits to (i) interaction contribution, (ii) table running sum columns, and (iii) each table's contribution:

  + For each table $Table_i$, populate the interaction contribution columns and compute the _table (LogUp) contribution_:

    + For each interaction $Interaction_j$ of table $Table_i$, initialize an empty _interaction contribution column_ of length $numRows_i$.

    + Initialise a _table running sum column_ $S_i in ExtensionField^(numRows_i)$ with the first value $S_i [0]$ populated according to the constraint choice.

    + *Constrain* the first row if required by selected constraint choice.

    + For each $j$-th row $R_j in BaseField^(numColumns_i)$ of $Table_i$, for $j in [numRows_i - 1]$:
      + For each $k$-th interaction $Interaction_k$ of table $Table_i$:
        + Compute the _interaction contribution numerator_ $ n_(j,k) = mu_(i,k) = w_(i,k)(R_j)[numElements_(i,k)] $
        + If $n eq.not 0$, compute the _interaction contribution denominator_ $ d_(j,k) = logupChallenge + fingerprintCoeff dot id_(i,k) + sum_(l = 0)^(numElements_(i,k) - 1) fingerprintCoeff^(l + 2) dot weightFunction_(i,k) (R_j)[l]. $
        + Save the _interaction contribution_ as $n_(j,k)/d_(j,k) in ExtensionField$ in the corresponding interaction contribution column for this interaction.
        + *Constrain* the interaction contribution column according to the definitions of $n$ and~$d$.

      + Compute the _row contribution_ as the sum $s_(j) = sum_k n_(j,k) / d_(j,k)$ and compute the next row's table running sum value $S_i [j+1] = S_i [j] + s_(j)$.

      + *Constrain* the transition of the running sum column as indicated by the constraint choice.

    + *Constrain* the last row if required by selected constraint choice.

  + Batch-commit to every table's interaction contribution columns and running sum columns with the column commitment scheme and commit to the table's overall contribution $S_i [N_i - 1]$ by sending it in the clear to the verifier.

+ Verifier checks that the sum of every table's overall contribution is equal to zero: $sum_i S_i [N_i - 1] = 0_ExtensionField$, and delegates the checks of the constraints to the STARK.

== Running Sum Constraint Choices <constraint_choices>

#cdsg[Write the constraints in this section more formally after STARK description has been written.]

=== Choice 1: transitions looking back

tl,dr: implicit $0_ExtensionField$ initial value, explicit final value.

+ (*Boundary, first row*) Constrain first row of running sum column to equal the sum of the first row of every interaction contribution column. (This is analogous an implicit $-1$-th row initialised at $0_ExtensionField$.)
+ (*Transition, looking back, applied to rows $1, dots, numRows_i - 1$*) For each row _other than the first_, constrain the _current_ running sum value to equal the sum of every current interaction contribution column added to the _previous_ running sum value.
+ (*Boundary, last row*) Constrain last row of running sum column to equal the claimed table contribution.

Total constraints: 2 boundary + 1 transition over $numRows_i - 1$ rows.

=== Choice 2: transitions looking forward

tl,dr: explicit $0_ExtensionField$ initial value, implicit final value.

+ (*Boundary, first row*) Constrain first row of running sum column to equal $0_ExtensionField$.
+ (*Transition, looking forward, applied to rows $0, dots, numRows_i - 2$*) For each row _other than the last_, constrain the _next_ running sum value to equal the sum of every current interaction contribution column added to the _current_ running sum value.
+ (*Boundary, last row*) Constrain last row of running sum column added to sum of last row of every interaction column to equal the claimed table contribution. (That is, the claimed table contribution is implicit in the last row of the table, but not written to last value of running sum column.)

Total constraints: 2 boundary + 1 transition over $numRows_i - 1$ rows.

=== Choice 3: circular transitions looking back/forward

+ For each row, constrain the _current/next_ (wrapping to first on last if "next") running sum value to equal the sum of every current interaction contribution value added to the _previous/current_ (wrapping to last on first if "previous") running sum value added to claimed table contribution divided by $numRows_i$.

Total constraints: 1 _circular_ transition over $numRows_i$ rows.

#aside("Justification")[
  This single circular constraint checks that each row's contribution $s_(i,j)$ is added to the running sum column, either in the current row's cell or in the next row's.
  In order to avoid boundary constraints, the look-back or peek-forward into the running sum column wraps around the beginning or end of the table.

  This alone implies that difference between first and last row's values will be the table's overall real contribution $sum_j s_(i,j)$, which will be incompatible with the circularity of the constraint.
  Since boundary constraints are avoided, the way to check that $sum_j s_(i,j)$ equals the claimed contribution $L_i$ is to remove a fraction of $L_i$ at each row in such a way that $L_i$ is removed completely after summing all $numRows_i$ rows; i.e., the constraint subtracts the public term $L_i / numRows_i$ from the running sum at every row.

  If the expected equality $sum_j s_(i,j) = L_i$ holds, then the circularity of the constraint will also hold.
]
