# Plan: rounds_2to4 a GPU (pendiente #1 del ledger)

Escrito 2026-07-27, sobre `gpu-opt-5090` @ d1c4f5ee+#14. Evidencia: análisis de código de esta fecha
(R2 propio; R3/R4 mapeados con file:line en este doc). Números de referencia: thread-sums del
profile 5tx (R2 evaluate 4.5s post-kernel-v2, decompose 5.5s, R3 OOD ~4s, R4 deep 5.7s,
FRI commit 2.9s + queries 1.7s) — re-rankear en continuations con los spans del tip antes de
implementar (ver §Protocolo).

Notación: N = lde_size (2^21 típico), 24N bytes = un codeword ext3 (~50MB).

## Hallazgos (con ubicación exacta)

### R2
- **[A1] Boundary-zerofier inverses** (`evaluator.rs:192-213`): vector LDE-size `v−g^step` +
  batch-inverse **en CPU, por constraint**. El dedup actual solo evita el `pow` del punto, no el
  vector: constraints que comparten step re-invierten lo mismo. El vector depende SOLO de
  (dominio, step) → idéntico entre constraints, tablas del mismo tamaño y TODOS los epochs.
  Encima se re-sube por PCIe en cada dispatch (`CompositionInputs.b_z_inv`, un slice LDE-size
  por constraint).
- **[A2] Trabajo domain-only recomputado por tabla×epoch**: `inv_2x` (batch-inverse de N/2,
  `prover.rs:1301-09`) y los weights del extend (`gpu_lde.rs:531-542`, N muls secuenciales).
  Hogar natural: `LdeTwiddles` (cache lazy por dominio que ya existe).
- **[A3] Round-trip PCIe del composition poly en el camino real (d=2)**: H sale del kernel GPU
  a host (24N D2H), decompose pointwise en CPU, mitades re-subidas (24N H2D), el extend GPU
  devuelve Vecs host SIN handle (`try_extend_two_halves_gpu`), el tree del commit re-sube
  (`try_build_comp_poly_tree_gpu` sobre slices host) y R4 re-sube las parts si el handle falta.
  **Todo el plumbing para evitarlo ya existe pero solo lo usa el branch d>2 que nunca corre**:
  `try_evaluate_parts_on_lde_gpu_keep` (prover.rs:1451), `set_gpu_composition_parts` (1513),
  consumers R4 (1808/1844), precedente fused keep-dev (`coset_lde_ext3_row_major_with_merkle_tree_keep_dev`).

### R3 (trace OOD ya está bien en GPU; los huecos:)
- **[B1] OOD de composition parts: SIEMPRE CPU** (`prover.rs:1546-1568`) — stride sobre Vecs
  host + barycentric CPU, sin intento de dispatch. Con el handle de A3 se cierra con el kernel
  `barycentric_ext3` que ya existe (mismo patrón que trace OOD, punto de eval `z^P`).
- **[B2] Domain-only por tabla×epoch**: `DomainConstants::from_domain` (`prover.rs:1544`,
  puntos del coset N + inversos) y `compute_frame_evaluation_points` (potencias de g)
  se recomputan siempre.
- **[B3] Invariantes recomputados K veces**: `z_pow_n` en el loop de eval points
  (`trace.rs:730` — (z·wᵉ)^N = z^N, constante) y el escalar final `ood_ext3_scalar`
  computado 2× por punto (main y aux: `gpu_lde.rs:1136/1215`).
- **[B4] Mismatch de thresholds**: `R3DevContext` se gatea a N·K ≥ 2^19 pero los kernels bary
  a N ≥ 2^14 → tablas medianas corren bary GPU con inv_denoms CPU + H2D por llamada.

### R4 (folds/trees/gathers ya residentes; los huecos:)
- **[C1] Round-trip DEEP→FRI**: el codeword DEEP baja (24N D2H, `deep.rs:244-262`), se
  bit-reversea en CPU (`prover.rs:1666-67`) y se re-sube entero (24N H2D, `fri.rs:78`).
  Fix: kernel de permutación bit-reverse on-device + `FriCommitState::new_dev` que tome el
  buffer device. — **[C1b] opcional**: los evals de cada layer FRI bajan (≈24N geométrico,
  `fri.rs:217-242`) solo para que el query phase lea `evaluation[index^1]` de host
  (`gpu_lde.rs:2096`); un gather device de los symmetric evals (patrón `gather_rows_*` que ya
  existe) eliminaría esa bajada.
- **[C2] Domain-only por prove**: FRI inverse twiddles (`compute_coset_twiddles_inv`, batch
  inverse de N/2, `fri_functions.rs:37-47`) + updates por layer + terminal offset —
  todos función pura de (coset_offset, domain_size), recomputados cada epoch.
- **[C3] El arm "mixed" del DEEP** construye e invierte denoms en CPU
  (`build_r4_inv_denoms_cpu`, batch inverse de N×(1+K)) cuando el path device falla —
  hoy no sabemos cuán seguido pasa (fallbacks silenciosos `.ok()`).

## Orden de implementación

Cada paso es un commit independiente, validable por separado. El criterio del orden:
dependencias primero, y dentro de eso, evidencia/esfuerzo.

> **Paso 0 corrido (2026-07-27, tip+paso1, 10tx cont, wall 13.8s)**: epoch_prove 34.0s thread
> → rounds_2to4 **21.6s (63%)**, r1_main_commit 8.1s, r1_aux 7.2s; trace build+collect 4.7s;
> global 3.6s. El blanco del plan confirmado en continuations. Contadores de fallback:
> pendientes, van con el paso 2.

| Paso | Qué | Toca | Riesgo | Expectativa* |
|---|---|---|---|---|
| 0 | ~~(server) Ranking de core-seconds en continuations~~ HECHO (ver arriba) + contadores de fallback [C3/D] pendientes | corrida + contadores triviales | nulo | dirige el resto |
| 1 | **Caches domain-invariant** [A2+B2+C2]: inv_2x y weights a `LdeTwiddles`; DomainConstants y frame-points por dominio; FRI inv-twiddles por (offset, size). + hoists [B3] | stark CPU-side | bajo (testable en mac) | varios core-s; wall chico |
| 2 | **Boundary zerofiers** [A1]: dedup por step + cache process-wide por (dims, step) + upload-once device-resident (cache device chico, ~16MB por (size,step)) | evaluator + gpu_interp/device layer | bajo-medio | parte de R2 evaluate + PCIe por dispatch |
| 3 | **Residencia d=2** [A3]: variante del kernel composition que deja H en device; kernel decompose pointwise (usa inv_2x cacheado, subido 1×); `try_extend_two_halves_gpu_keep` (dev-in, dev-out + handle); tree desde device; `set_gpu_composition_parts` en el camino d=2 | math-cuda (2 kernels chicos + variantes) + prover.rs | medio | mata ~4-5×24N PCIe/tabla/epoch + decompose host (5.5s thread) |
| 4 | **DEEP→FRI residente** [C1]: bit-reverse kernel + `FriCommitState::new_dev` | math-cuda + fri | medio | 2×24N PCIe/tabla/epoch |
| 5 | **Comp-parts OOD en GPU** [B1] (necesita el handle del paso 3) | prover.rs R3 + bary existente | bajo | parte del R3 OOD 4s |
| 6 | **Threshold alignment** [B4] + revisar contadores del paso 0 [C3] | gates | bajo | tablas medianas |
| 7 | (opcional, según datos) [C1b] symmetric evals por gather device | fri + merkle | medio | ~24N/tabla/epoch |

*Las expectativas son thread-time del profile 5tx; la lección de la campaña es que solo mueven
wall los que atacan decenas de core-s o el schedule — por eso los pasos 3+4 (PCIe + host passes
estructurales) son el corazón del plan, y el paso 1 va primero solo porque es barato y
des-ruidea las mediciones siguientes.

## Protocolo de validación (por paso)

1. Local (mac): `cargo test -p stark`, `-p lambda-vm-prover` (515/522 esperado), clippy.
2. Server: build cuda, suite `cargo test -p stark --features cuda`, cross-verify bidireccional
   (proof nuevo × verifier viejo y viceversa, 10tx cont).
3. ABBA 6-8 pares vs el commit anterior (10tx cont, e20@K3 para comparabilidad con la campaña).
4. Los cambios de pura residencia/cache no cambian la matemática → si ABBA da regresión,
   es bug de scheduling/VRAM, no "tradeoff".
5. VRAM: el paso 3 deja las parts (48MB/tabla @2^21) residentes R2→R4 — dentro de un prove las
   tablas van secuenciales, así que el pico agregado es ~1-2 tablas; verificar con K=3 igual.

## Riesgos y guardas

- **Device-only mode**: los fallbacks a CPU asertan si el host trace está vacío — todo nuevo
  `None`/`.ok()` debe mantener el patrón de snapshot/restore del transcript (FRI ya lo hace).
- **Soundness**: nada de esto toca la matemática del protocolo; los caches son de valores
  deterministas del dominio. El equality-check del cache de lowering (#14) es el patrón a
  seguir donde un cache pueda aliasear.
- **Proof bytes**: ya son no-deterministas run-to-run (preexistente), así que la validación es
  cross-verify, no byte-compare.
- **Tablas chicas** (LDE < 2^19): mantienen el camino CPU actual — los caches del paso 1-2
  las benefician igual.
