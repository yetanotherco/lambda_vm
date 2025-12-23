# Description

The different components that form the pipeline for proving the correctness of the execution of a given program on an input stream are:
1. The source code of the program, written in high-level language
2. The program binary, ready for the virtual machine
3. The execution record of the binary over the VM architecture for a given input
4. The witness of the computation, generated from the execution record. Typically, this will consist of several tables, called trace tables.
5. The proof of validity of the witness for some language and VM architecture

The steps are as follows:
1. The *compiler* transforms the program into the binary.
2. The *executor* takes a binary, an input stream and an VM architecture and produces the execution record.
3. The *witness generator* transforms the execution record into a witness compatible with the chosen arithmetisation and constraint system.
4. The *proof system* takes the witness and the constraint system, and produces a (set of) proof(s) that the former satisfies the latter.