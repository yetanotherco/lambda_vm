# Description of the proof system

The proof system is the component responsible for generating the certificate of computational integrity and determines the efficiency and key properties of the virtual machine and proof. Among them, we want:
1. Aim at least for 100 bits of provable security.
2. Have a transparent setup.
3. Ensure that the proof system is post-quantum secure.
4. Have as few cryptographic primitives and assumptions as possible.
5. Have short proofs.

This section will cover the basic cryptographic primitives needed for the proof system and a description of the whole proof system and arguments used. Core concepts are:

> **Note:** the chapters below are a work in progress.

1. Finite field
2. Polynomials
3. Extension field
4. Hash function
5. Fast-Fourier transform
6. Reed-Solomon codes
7. Constraint
8. Algebraic intermediate representation
9. Interactive oracle proof
10. Fast Reed-Solomon Interactive Oracle Proof of Proximity (FRI)
11. Provable security and conjectured security
12. Lookup argument

The flow of the proof system is described in the following section. 