### wt1202-tree.log

log `/root/prof/wt1202-tree.log` · kind **whir**
 · ZF FORMAT `cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query` (1 line(s))
 · tag wt1202 expect 9e2728955 head 9e272895


| field | value |
|---|---|
| block wall (WHOLE RUN) | 40.4 s |
| host peak (WHOLE RUN) | 15.881 GiB |
| max RSS (/usr/bin/time) | 15.876 GiB |
| base | 30.8 s (15 epochs) |
| level 0 | NA s |
| interior (pooled) | NA s (levels NA, NA nodes) |
| root stage · verify | 1.2 s · 0.1 s · 180 words |
| device peak (10 ms sampler) | 27,570 MiB · base window 27,570 · after 25,522 · reserved high-water 24,342 |
| fallbacks commit / device | 0 / 0 |
| PROVED AND VERIFIED (root) | yes |
| test result | ok (1 passed, 0 failed) |
| SHAPE | SHAPE from 15 epochs at fan-in 5: 2 levels, 4 nodes |
| global wrap prove | 2.0 s |
| GRIND KNOBS banner | yes |

**Census per level** (perms = `LFM_HASH` real rows; rows = its committed height)

| level | proofs | cells | instructions | LFM_HASH perms | LFM_HASH rows | proof bytes |
|---|---|---|---|---|---|---|
| global | 1 | 238,421,440 | 3,227,416 | 268,706 | 524,288 | NA |
| L1 | 3 | 852,638,976 | 16,246,874 | 1,318,891 | 1,572,864 | NA |
| root | 1 | 149,556,992 | 2,862,759 | 253,082 | 262,144 | NA |
| Σ censused | 5 | 1,240,617,408 | 22,337,049 | 1,840,679 | | |

<details><summary>per wrap / per node</summary>

| proof | level | cells | instructions | LFM_HASH real/rows | program_id |
|---|---|---|---|---|---|
| L1N1 (arity 5) | L1 | 290,854,656 | 5,632,652 | 450,265/524,288 | 43c471b4e2c21ba9 |
| L1N0 (arity 5) | L1 | 290,852,608 | 5,355,197 | 445,665/524,288 | ab139aaa74a52e7c |
| the WHIR GLOBAL wrap | global | 238,421,440 | 3,227,416 | 268,706/524,288 | a3f2833fe8c11ecf |
| L1N2 (arity 5) | L1 | 270,931,712 | 5,259,025 | 422,961/524,288 | 3c46045c0e1dcbab |
| the WHIR BLOCK-ARTIFACT ROOT | root | 149,556,992 | 2,862,759 | 253,082/262,144 | 987a2780df8324ca |

</details>

identities: 0 wraps md5 `None` · 3 nodes md5 `051f03969465` · global `a3f2833fe8c11ecf` · root `987a2780df8324ca`


SUMMARY rc=0 (all logs complete)
