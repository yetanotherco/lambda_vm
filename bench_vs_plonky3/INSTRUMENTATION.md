# `bench_vs_plonky3` — puntos de instrumentación

Guía de referencia para revisores / handoff. Describe **dónde está cada timer
y qué mide** en la comparación Lambda STARK vs Plonky3. No describe el AIR
en sí (eso vive en `ANALYSIS_LOG.md`).

## Cómo correrlo

El test que imprime el breakdown se llama `instruments_breakdown`. Hay que
compilar con la feature `instruments` y pasar `--nocapture` porque la salida
va a stdout (si no, `cargo test` se la come).

**x86 (Goldilocks scalar, SSE2 Keccak residual en P3):**

```bash
RUSTFLAGS="-C target-feature=-avx2,-avx512f" \
cargo test -p bench-vs-plonky3 --features instruments --release -- \
  instruments_breakdown --nocapture
```

## Entrada principal

- Archivo: `bench_vs_plonky3/src/lib.rs`
- Función: `instruments_breakdown` (línea 82)
- AIR Fibonacci fijo:
  - `num_sequences = 16`
  - `rows = 1 << 18` (2^18)
  - columns = 32 (2 por secuencia)
  - `blowup_factor = 2`
  - `fri_number_of_queries = 219`
  - `grinding_factor = 0`

El test hace dos pasadas independientes:

1. Corre Lambda STARK con los timers internos del crate `stark` (feature
   `instruments`).
2. Corre Plonky3 con un `tracing_subscriber` custom que captura spans.

## Feature flags

`bench_vs_plonky3/Cargo.toml` (líneas 33-40):

```toml
[features]
default    = ["parallel"]
parallel   = ["stark/parallel"]
instruments = ["stark/instruments"]
```

`crypto/stark/Cargo.toml` (líneas 35-41):

```toml
[features]
instruments = []                       # prints de timing en prover/verifier
parallel    = ["dep:rayon", "crypto/parallel"]
```

`instruments` y `parallel` **coexisten** (no son excluyentes). En la práctica
los benchmarks corren siempre con ambos activos: Plonky3 usa
`Radix2DitParallel` (rayon) unconditionally, así que Lambda también tiene que
correr en paralelo para comparar apples-to-apples.

## Lambda: estructuras de timing

`crypto/stark/src/instruments.rs`.

### `MultiProveTiming` (líneas 40-50)

Recolectada dentro de `multi_prove` y consumida por el test vía
`stark::instruments::take()`.

| Campo | Qué mide |
|---|---|
| `prepass` | Construcción de domains + `LdeTwiddles` caches. |
| `main_commits` | Round 1 Phase A: commit de todos los main traces. |
| `aux_build` | Round 1 Phase B: construcción de aux traces / LogUp. |
| `aux_commit` | Round 1 Phase B: LDE + Merkle commit de aux traces. |
| `rounds_2_4` | Tiempo total de Rounds 2-4 (todas las tablas). |
| `round1_sub` | Sub-op breakdown de Round 1 (`Round1SubOps`). |
| `table_timings` | Por tabla: `(name, rows, duration, TableSubOps)`. |

### `Round1SubOps` (líneas 28-37)

Sub-ops dentro de Round 1. Se acumulan en `AtomicU64`, así que workers rayon
las pueden incrementar en paralelo sin perder datos.

| Campo | Qué mide |
|---|---|
| `main_lde` | Main trace: `expand_columns_to_lde` (LDE/FFT). |
| `main_merkle` | Main trace: `commit_columns_bit_reversed` (Merkle). |
| `aux_lde` | Aux trace: `expand_columns_to_lde`. |
| `aux_merkle` | Aux trace: `commit_columns_bit_reversed`. |

### `TableSubOps` (líneas 7-24)

Por tabla, dentro de Rounds 2-4. Las partes de R2/R4 se pasan por
thread-locals (`R2_SUB`, `R4_SUB`) y después se ensamblan en
`prove_rounds_2_to_4` (ver más abajo).

| Campo | Round | Qué mide |
|---|---|---|
| `constraints` | R2 | `evaluator.evaluate()` — constraints sobre dominio LDE. |
| `comp_decompose` | R2 | `decompose_and_extend_d2` — iFFT + extensión del composition poly. |
| `comp_commit` | R2 | Merkle commit del composition poly. |
| `ood` | R3 | Barycentric OOD eval (ver nota sobre dónde se captura). |
| `deep_comp` | R4 | `compute_deep_composition_poly_evaluations`. |
| `deep_extend` | R4 | `interpolate_fft` + `evaluate_fft` para extender el deep comp poly. |
| `fri_commit` | R4 | `fri::commit_phase_from_evaluations` (folds + Merkle layers). |
| `queries` | R4 | Grinding (si hay) + sampling + FRI query phase + Merkle openings. |

### Dónde se capturan (en `crypto/stark/src/prover.rs`)

- `multi_prove` (línea 1490):
  - `reset_all()` (1502).
  - `prepass` timer (1515-1533).
  - `main_commits` timer (1541-…).
  - `aux_build`, `aux_commit` timers (durante Round 1 Phase B).
  - `rounds_2_4` timer; al final: `store(MultiProveTiming)`.
- `round_2_compute_composition_polynomial` — `constraints` / `comp_decompose` /
  `comp_commit` (vía `store_r2_sub`).
- `prove_rounds_2_to_4` — **acá** se captura el OOD:
  `round_3_dur = t_r3.elapsed()` en líneas 1957-1967, y se guarda en
  `TableSubOps.ood` (línea 2010). `round_3_evaluate_polynomials_in_out_of_domain_element`
  **no** tiene instrumentación propia.
- `round_4_compute_and_run_fri_on_the_deep_composition_polynomial` —
  `deep_comp` / `deep_extend` / `fri_commit` / `queries`
  (vía `store_r4_sub`).

## Plonky3: breakdown por spans

Todo vive dentro de `instruments_breakdown` en `bench_vs_plonky3/src/lib.rs`,
después del bloque de Lambda.

- Se define una `P3TimingLayer` custom (líneas 216-259) que implementa
  `tracing_subscriber::Layer`:
  - `on_new_span` guarda el nombre del span.
  - `on_enter` guarda `Instant::now()`.
  - `on_close` calcula `start.elapsed()` y lo empuja a un `Vec<(name, ms)>`.
- Se monta un subscriber con `LevelFilter::DEBUG` (línea 266) y se instala
  como default **sólo durante el `p3_uni_stark::prove`** (líneas 275-280,
  scope con `_guard`).
- Post-prove: orden descendente por duración (287), filtra spans con
  `ms >= 0.1` (289), y calcula `(unaccounted) = total − Σspans` (293-301).

### Qué implica el diseño

- **La capa no filtra por crate**: captura *cualquier* span DEBUG emitido
  mientras el subscriber está vivo. En la práctica sólo corre
  `p3_uni_stark::prove` dentro de ese bloque, así que todos los spans que
  salen son de Plonky3 — pero si alguien agrega un `#[instrument]` propio
  dentro del scope del guard, también se va a contar.
- **No hay instrumentación manual de funciones de Plonky3.** La granularidad
  del breakdown = spans que Plonky3 ya emite internamente.
- **Nesting / doble-conteo:** P3 tiene spans anidados (p.ej.
  `prove ⊃ compute_quotient_values ⊃ evaluate_constraints`). Cada span se
  cuenta una vez con su wall-clock entre `on_enter` y `on_close`, así que
  **`Σspans > wall-clock` es esperable, no es un bug**. Consecuencia:
  `(unaccounted) = total − Σspans` **puede quedar negativo** en presencia de
  nesting — no significa que falte tiempo, significa que los spans padre se
  solapan con sus hijos. El código sólo imprime `(unaccounted)` si
  `> 1.0ms`, así que casos negativos se silencian.

## Segunda capa de instrumentación (no la usa `bench_vs_plonky3`)

Existe una capa adicional en `prover/src/instruments.rs` (líneas 54-211,
`print_report`) — orientada al ejecutor del VM (execute + trace build + AIR
construction) que además re-imprime el `MultiProveTiming` del STARK con
otro formato. `bench_vs_plonky3` **no** la invoca; sólo consume
`stark::instruments::take()` directamente. Vale la pena saberlo si buscás
timings y aparecen en logs distintos.

## Advertencias para el revisor

1. Lambda: timing manual, específico del pipeline `multi_prove`. Granularidad
   fina pero acoplada al código — moverlo rompe los breakpoints.
2. Plonky3: span-based. Granularidad = la que P3 decida exponer. Si P3 deja
   de emitir un span en una versión futura, la línea desaparece del reporte
   sin previo aviso.
3. Los porcentajes de Lambda se calculan contra el **total wall-clock del
   test** (no contra `rounds_2_4`), así que la suma no cierra al 100% — hay
   tiempo fuera de `multi_prove` (construcción de AIR, setup).
4. Los porcentajes de Plonky3 se calculan contra **`p3_prove_dur`** (solo el
   `prove`, sin setup).
5. El benchmark usa **degree 3** para la extensión de Plonky3 vía git deps a
   la rama `feat/goldilocks_deg3` del fork `yetanotherco/Plonky3` (ver
   `bench_vs_plonky3/Cargo.toml`), que provee `BinomiallyExtendable<3>`
   para Goldilocks con el mismo irreducible `x^3 - 2` que Lambda.
6. Plataforma: x86 con `RUSTFLAGS="-C target-feature=-avx2,-avx512f"` →
   Goldilocks scalar, residual SSE2 en Keccak de P3 (~7%).
