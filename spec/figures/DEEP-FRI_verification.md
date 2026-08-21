<!-- Use the following code on sequencediagram.org to modify the picture -->

title Proof verification
note over P,V: established: shared program with\npublic input
note over P: fill tables
P->V: batch-commit tables
group par [DEEP]
P<-V: lincomb challenges
P->V: quotient commitment
P<-V: segment challenges
P->V: segment commitment
P<-V: DEEP point
P->V: DEEP openings
else LogUp
P<-V: LogUp challenges
P->V: batch-commit to LogUp columns
P->V: open sum entries
note over V: checksum
end

P->V: batch FRI-commit (implicitly)
loop Batch-FRI
P<-V: folding challenge
P->V: folding commitment
end
P->V: FRI low-degree output
note over V: verify low-degreeness
loop FRI-verify
note over V: sample folding\nchallenges
P<-V: challenges
P->V: openings
note over V: verify openings
note over V: verify folding\nsteps
end
note over V: verify DEEP