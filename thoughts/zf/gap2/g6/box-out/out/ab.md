### wt1200 — arm P1 (no ZF knob)

log `/root/prof/wt1200-tree.log` · kind **whir**
 · ZF FORMAT `cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query` (1 line(s))
 · tag wt1200 expect 9e2728955 head 9e272895


| field | value |
|---|---|
| block wall (WHOLE RUN) | 39.8 s |
| host peak (WHOLE RUN) | 16.172 GiB |
| max RSS (/usr/bin/time) | 16.166 GiB |
| base | 30.4 s (15 epochs) |
| level 0 | NA s |
| interior (pooled) | NA s (levels NA, NA nodes) |
| root stage · verify | 1.2 s · 0.1 s · 180 words |
| device peak (10 ms sampler) | 27,634 MiB · base window 27,634 · after 25,586 · reserved high-water 24,342 |
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
| L1N0 (arity 5) | L1 | 290,852,608 | 5,355,197 | 445,665/524,288 | ab139aaa74a52e7c |
| L1N1 (arity 5) | L1 | 290,854,656 | 5,632,652 | 450,265/524,288 | 43c471b4e2c21ba9 |
| the WHIR GLOBAL wrap | global | 238,421,440 | 3,227,416 | 268,706/524,288 | a3f2833fe8c11ecf |
| L1N2 (arity 5) | L1 | 270,931,712 | 5,259,025 | 422,961/524,288 | 3c46045c0e1dcbab |
| the WHIR BLOCK-ARTIFACT ROOT | root | 149,556,992 | 2,862,759 | 253,082/262,144 | 987a2780df8324ca |

</details>

identities: 0 wraps md5 `None` · 3 nodes md5 `051f03969465` · global `a3f2833fe8c11ecf` · root `987a2780df8324ca`

### wt1201 — arm N (nsys)

log `/root/prof/wt1201-tree.log` · kind **whir**
 · ZF FORMAT `cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query` (1 line(s))
 · tag wt1201 expect 9e2728955 head 9e272895


| field | value |
|---|---|
| block wall (WHOLE RUN) | 41.0 s |
| host peak (WHOLE RUN) | 17.010 GiB |
| max RSS (/usr/bin/time) | 0.064 GiB |
| base | 31.5 s (15 epochs) |
| level 0 | NA s |
| interior (pooled) | NA s (levels NA, NA nodes) |
| root stage · verify | 1.2 s · 0.1 s · 180 words |
| device peak (10 ms sampler) | 27,440 MiB · base window 27,440 · after 25,668 · reserved high-water 24,342 |
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
| L1N0 (arity 5) | L1 | 290,852,608 | 5,355,197 | 445,665/524,288 | ab139aaa74a52e7c |
| L1N1 (arity 5) | L1 | 290,854,656 | 5,632,652 | 450,265/524,288 | 43c471b4e2c21ba9 |
| the WHIR GLOBAL wrap | global | 238,421,440 | 3,227,416 | 268,706/524,288 | a3f2833fe8c11ecf |
| L1N2 (arity 5) | L1 | 270,931,712 | 5,259,025 | 422,961/524,288 | 3c46045c0e1dcbab |
| the WHIR BLOCK-ARTIFACT ROOT | root | 149,556,992 | 2,862,759 | 253,082/262,144 | 987a2780df8324ca |

</details>

identities: 0 wraps md5 `None` · 3 nodes md5 `051f03969465` · global `a3f2833fe8c11ecf` · root `987a2780df8324ca`

### wt1202 — arm P2 (no ZF knob)

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

### wt1203 — arm C (LAMBDA_VM_ROW_CENSUS=1)

log `/root/prof/wt1203-tree.log` · kind **whir**
 · ZF FORMAT `cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query` (1 line(s))
 · tag wt1203 expect fd6da62a2 head fd6da62a


| field | value |
|---|---|
| block wall (WHOLE RUN) | 40.3 s |
| host peak (WHOLE RUN) | 16.268 GiB |
| max RSS (/usr/bin/time) | 16.263 GiB |
| base | 30.6 s (15 epochs) |
| level 0 | NA s |
| interior (pooled) | NA s (levels NA, NA nodes) |
| root stage · verify | 1.2 s · 0.1 s · 180 words |
| device peak (10 ms sampler) | 27,794 MiB · base window 27,794 · after 25,554 · reserved high-water 24,342 |
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
| L1N0 (arity 5) | L1 | 290,852,608 | 5,355,197 | 445,665/524,288 | ab139aaa74a52e7c |
| L1N1 (arity 5) | L1 | 290,854,656 | 5,632,652 | 450,265/524,288 | 43c471b4e2c21ba9 |
| the WHIR GLOBAL wrap | global | 238,421,440 | 3,227,416 | 268,706/524,288 | a3f2833fe8c11ecf |
| L1N2 (arity 5) | L1 | 270,931,712 | 5,259,025 | 422,961/524,288 | 3c46045c0e1dcbab |
| the WHIR BLOCK-ARTIFACT ROOT | root | 149,556,992 | 2,862,759 | 253,082/262,144 | 987a2780df8324ca |

</details>

identities: 0 wraps md5 `None` · 3 nodes md5 `051f03969465` · global `a3f2833fe8c11ecf` · root `987a2780df8324ca`


## A/B

| tag/log | arm | knobs | ZF banner | wall s | host GiB | base | L0 | interior | root | dev MiB | fb c/d | proved | wraps md5 | nodes md5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wt1200 | P1 | whir:- | cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query | 39.8 | 16.17 | 30.4 | NA | NA | 1.2 | 27,634 | 0/0 | yes | None | 051f03969465 |
| wt1201 | N | whir:nsys | cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query | 41.0 | 17.01 | 31.5 | NA | NA | 1.2 | 27,440 | 0/0 | yes | None | 051f03969465 |
| wt1202 | P2 | whir:- | cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query | 40.4 | 15.88 | 30.8 | NA | NA | 1.2 | 27,570 | 0/0 | yes | None | 051f03969465 |
| wt1203 | C | whir:LAMBDA_VM_ROW_CENSUS=1 | cap=auto whir_cap=auto fri=dp one_row=0 whir_folds=first6 whir_stack=27 whir_grind=query | 40.3 | 16.27 | 30.6 | NA | NA | 1.2 | 27,794 | 0/0 | yes | None | 051f03969465 |

| knob setting | arms | walls s | mean s | spread s | Δ mean vs first setting | same-setting identities | same-setting census |
|---|---|---|---|---|---|---|---|
| whir:- | 2 | 39.8, 40.4 | 40.10 | 0.60 | +0.00 | identical | identical |
| whir:nsys | 1 | 41.0 | 41.00 | NA | +0.90 | n=1 | n=1 |
| whir:LAMBDA_VM_ROW_CENSUS=1 | 1 | 40.3 | 40.30 | NA | +0.20 | n=1 | n=1 |

identities `whir:-` vs `whir:nsys`: IDENTICAL (a format change that moves no program id? read before quoting)

identities `whir:-` vs `whir:LAMBDA_VM_ROW_CENSUS=1`: IDENTICAL (a format change that moves no program id? read before quoting)

**Census per level per arm** (cells / instructions / LFM_HASH perms)

| tag/log | knobs | global | L1 | root | Σ perms |
|---|---|---|---|---|---|
| wt1200 | whir:- | 238,421,440 / 3,227,416 / 268,706 | 852,638,976 / 16,246,874 / 1,318,891 | 149,556,992 / 2,862,759 / 253,082 | 1,840,679 |
| wt1201 | whir:nsys | 238,421,440 / 3,227,416 / 268,706 | 852,638,976 / 16,246,874 / 1,318,891 | 149,556,992 / 2,862,759 / 253,082 | 1,840,679 |
| wt1202 | whir:- | 238,421,440 / 3,227,416 / 268,706 | 852,638,976 / 16,246,874 / 1,318,891 | 149,556,992 / 2,862,759 / 253,082 | 1,840,679 |
| wt1203 | whir:LAMBDA_VM_ROW_CENSUS=1 | 238,421,440 / 3,227,416 / 268,706 | 852,638,976 / 16,246,874 / 1,318,891 | 149,556,992 / 2,862,759 / 253,082 | 1,840,679 |

**Census delta per level, first arm of each setting vs the first setting** (perms, ratio)

- `whir:nsys`: global +0 (1.000×) · L1 +0 (1.000×) · root +0 (1.000×)
- `whir:LAMBDA_VM_ROW_CENSUS=1`: global +0 (1.000×) · L1 +0 (1.000×) · root +0 (1.000×)

A/B identity check within settings: OK

A/B census check within settings: OK

SUMMARY rc=0 (all logs complete)
