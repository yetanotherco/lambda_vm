#import "/book.typ": book-page, et

#show: book-page("verifier")

= Definitions
#et("Provide definitions for the notation used in this chapter. Perhaps this can be shared with some of the other theoretical chapters.")


= Proof contents
#et("outline the contents of the proof sent by the prover. This can then be used to reference what exactly needs to be checked by the verifier.")
Let $T_i$ be the $i$ tables of the VM, each with columns $c_(i,j)$.

The proof contains the following:

+ merkle commitments $#`CC`_i$ to the evaluations of $f_i$ (polynomials that evaluates to columns $c_i$ on $<g>$) over the FRI-domain.
+ merkle commitments $"SC"_i$ to the evaluations of the segments over the FRI-domain.
+ merkle commitment to the "DEEP composition polynomial" (DCP)


= Proof verification
#et("outline the steps that need to be taken by the verifier to verify a proof.")

The verifier must perform the following checks to ensure correctness of a proof:

#set enum(numbering: "i.1.a.", body-indent: 1em)


+ *Memory initialization/finalization correctness*:
    + all `page` numbers are multiples of the page length,
    + all `page` numbers are used as most once,
    + all registers are initialized as $0$,
    + all registers have $0$ as final value.

+ *Table correctness (FRI)*:
    + verify consistency between $"SC"_i$, $"CC"_i$ and the constraints:
        + TODO
    + verify consistency between SC and DCP

    + reconstruct challenges:
        + #et("figure out how this works")
    + verify constraints:
        + arithmetic constraints:
            - divisor
        + interaction constraints:
            - nothing
        + template constraints:
            - apply divisors for arithmetic constraints
    + Verify the `HALT` execution.


+ *Table interaction correctness (LogUp)*: #et("tag logup once merged")
    + test