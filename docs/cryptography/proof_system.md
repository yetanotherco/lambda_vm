# Description of the proof system

The proof system is the component responsible for generating the certificate of computational integrity and determines the efficiency and key properties of the virtual machine and proof. Among them, we want:
1. Aim at least for 100 bits of provable security.
2. Have a transparent setup.
3. Ensure that the proof system is post-quantum secure.
4. Have as few cryptographic primitives and assumptions as possible.
5. Have short proofs.

This section will cover the basic cryptographic primitives needed for the proof system and a description of the whole proof system and arguments used. Core concepts are:
1. [Finite field](./finite_field.md)
2. [Polynomials](./polynomials.md)
3. [Extension field](./extension_field.md)
4. [Hash function](./hash_function.md)
5. [Fast-Fourier transform](./fast_fourier_transform.md)
6. [Reed-Solomon codes](./reed_solomon_codes.md)
7. [Constraint](./constraint.md)
8. [Algebraic intermediate representation](./air.md)
9. [Interactive oracle proof](./iop.md)
10. [Fast Reed-Solomon Interactive Oracle Proof of Proximity (FRI)](./fri.md)
11. [Provable security and conjectured security](./security.md)
12. [Lookup argument](./lookup.md)

The flow of the proof system is described in the following section. 