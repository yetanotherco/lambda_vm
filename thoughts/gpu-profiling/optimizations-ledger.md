# Ledger de optimizaciones del prover (actualizado 2026-07-24)

Workload de referencia: ethrex 10tx continuations (`executor/tests/ethrex_10_transfers.bin`, `--continuations`).
Máquina objetivo: lambdavm-5090-01, 16 cores + RTX 5090 (lambdavm-5090-01).
**Headline: main 22.52s → gpu-opt-5090 13.88s = 1.62× (−38%), proofs compatibles con el verifier de main.**
**Update 2026-07-27 final (items 14-19 commiteados en gpu-opt-5090, 7 commits locales sobre 7f65014f): HEADLINE OFICIAL medido PR-vs-main 12 pares: 22.58 → 11.05s = 2.04× (−51.1%, sd 0.88%). Gates: suite cuda 212/212, cross-verify PR↔main ambas direcciones, clippy cuda 0, stress anti-hang 20/20.**
**Bug encontrado y arreglado por los gates (commit 827919db): deadlock intermitente (~1/25) — batch-inverse PARALELO dentro de OnceLock init de los caches de dominio compartidos vs pool de rayon (workers bloqueados en la celda matan de hambre al inicializador). Fix: inversores secuenciales + prefill en domain_and_twiddles. REGLA: nunca trabajo rayon adentro de un lazy-init compartido.**
**Nota post-mortem: matar proves CUDA colgados requiere SIGKILL (timeout -k) y verificar nvidia-smi compute-apps — zombies reteniendo VRAM causan cascadas de fallas espurias en corridas siguientes.**
**Review multi-agente 2026-07-27 (5 lentes: soundness, concurrencia, CUDA, unsafe, fallbacks): soundness/protocolo LIMPIO (paridad CPU↔GPU verificada punto a punto), sin UB alcanzable en éxito. 2 bugs reales arreglados (commits de0a3180/ac37c3fa): pipeline de epochs se colgaba para siempre ante un Err mid-run (senders bounded sin consumidor), y PendingD2H sin Drop sincronizante (UAF de slab pinned en error paths). + hardening (doble-build pin, asserts de shapes, gates). FOLLOW-UPS pendientes (issues aparte): telemetría en fronteras .ok() del chain residente, doble conteo GPU_* en retries, caches sin evicción para servicios multi-programa, admisión de VRAM cross-prove con K>1.**

## ✅ HECHAS (en `gpu-opt-5090`, 8 commits + spans)

| # | Optimización | Efecto medido | Commit |
|---|---|---|---|
| 1 | Async pinned D2H (cuMemcpyDtoHAsync + slabs pinned por worker + evento reusable) | DtoH host-block 12.4→6.7s; wall neutro en 48c (paralelismo ya lo tapaba) | 530f423f |
| 2 | Ready-events device-side (consumers cuStreamWaitEvent en vez de sync del productor) | ídem, estructural | 530f423f |
| 3 | Pool de eventos pre-creados en init (cuEventCreate mid-prove convoya el driver lock ~30ms) | ídem | 530f423f |
| 4 | Cache process-wide de árboles Merkle precomputados (key = commitment root) | parte del 16.1→14.5s | 530f423f |
| 5 | Pipeline de epochs (producer thread ejecuta/construye epoch i+1 mientras prueba i) | 24.0→21.0s (48c) | f102818b |
| 6 | K provers concurrentes (LAMBDA_VM_EPOCH_CONCURRENCY=3) | 21.0→16.1s (48c) | f102818b |
| 7 | DECODE commitment 1× por prove (antes por epoch) | incluido arriba | f102818b |
| 8 | Kernel constraints dim-split + liveness slots + uniform propagation (v2) | kernel 814→267ms (3×); scratch 8-35× menor (mata OOM-fallback silencioso); wall −2.6% | 58cb87ef |
| 9 | Merkle de tablas preprocessed (DECODE/BITWISE) por pipeline GPU fused + residencia R2-R4 | wall −3.7% (32c) | 057d68c1 |
| 10 | Prove global solapado con la cola de epochs (boundaries por canal, hilo scoped) | −3.5%; cola serial 0.9s eliminada (Gantt) | 966d3356 |
| 11 | Cache DECODE artifacts por ELF (B2) | neutro (~0.5 core-s de ~580); base de A | 0098e946 |
| 12 | Trace builds a pool de builders; boundary/fini sin trace (A) | cadena serial 7.2→2.9s; wall neutro en 16-32c (CPU-bound), pagaría con más cores | 78d18b91 |
| 13 | Cache de prototipos AIR pre-capturados por (tabla, opts) (B1) | neutro (~1-2 core-s); acelera monolítico/tests | 7c54af74 |
| 14 | Cache del lowering del constraint program (content-hash + eq-check) + IR capturado compartido por Arc entre clones | ABBA 8 pares: **−1.91%** (14.29→14.02s, 7/8 pares, p≈0.02); cross-verify OK ambas direcciones | 7f65014f |
| 15 | Paso 1 del plan rounds_2to4: cache process-wide Domain+LdeTwiddles + inv_2x/DomainConstants/FRI-inv-twiddles como OnceLock por dominio | ABBA 8 pares: **−1.83%** (14.19→13.92s, 7/8, p≈0.03); suite cuda 212/212; cross-verify OK | (staged 2026-07-27, sin commitear) |
| 16 | Paso 2a: boundary-zerofier inverses cacheados por (dominio, step) con Arc compartido (antes: batch-inversion LDE-size POR CONSTRAINT por tabla por epoch) | ABBA 8 pares: **−2.15% mean / −1.80% median** (13.78→13.48s, 6/8, sd alta 3%); suite cuda 212/212; cross-verify OK | (staged 2026-07-27, junto a #15) |
| 17 | Paso 2b: b_z_inv device-resident (GpuBaseVec upload-once + D2D al buffer flat; cache keyed por Arc ptr, anti-ABA pineando el Arc) | **neutro en wall** a 10tx e20@K3 (ABBA 8p: +0.95%/−0.06% median, sd 3.95% — box ruidoso); estructural: mata ~32-96MB PCIe/tabla/epoch, patrón base del paso 3 | (staged 2026-07-27) |
| 18 | **Paso 3: residencia d=2** — H queda en device (`eval_composition_on_device_keep`), kernel `decompose_d2_ext3` pointwise a slabs, `coset_lde_batch_ext3_slabs_keep` (butterflies sin H2D), handle de parts a R4 DEEP; fallbacks en cascada (download H → host path) | ABBA 8 pares: **−15.59% / −15.82% median (13.48→11.37s, 8/8, p<1e-4)**; suite cuda 212/212; cross-verify OK. **Main 22.52 → 11.37s = 1.98× solo código.** | (staged 2026-07-27) |
| 19 | **Paso 4: DEEP→FRI residente** — DEEP `_keep` con bit-reverse on-device (kernel `bit_reverse_ext3_interleaved`) + `FriCommitState::new_dev` adopta el codeword como buffer de fold (cero H2D de evals); loop del commit extraído y compartido host/dev con snapshot/restore del transcript | ABBA 8 pares: **−2.70% / −2.65% median (11.25→10.95s, 7/8, p≈0.008)**; suite cuda 212/212; cross-verify OK. **Main 22.52 → 10.95s = 2.06× solo código.** | (staged 2026-07-27) |

Lección clave: el prove consume ~580 core-seconds; optimizaciones de <5 core-s no mueven el wall en boxes CPU-bound. Los pasos 5-6-8-9-10 movieron el wall porque atacaron decenas de core-seconds o el schedule crítico.

## ❌ DESCARTADAS / APARCADAS (con evidencia)

- **K=4**: empata con K=3 en 48c y en 32c (techo = CPU, VRAM sobra). Re-evaluar solo si bajan mucho los core-seconds.
- **Pre-size hint de pinned slabs**: REGRESIONÓ (7.5s de HostAlloc — cada worker alocaba el máximo global). Revertida; rediseño solo si el churn (~2s) molesta.
- **Waits diferidos cross-función (Fase B-iii/C de opt#2)**: bajo ROI medido.
- **MAX_THREADS del kernel interp** (subirlo con el scratch chico): kernel ya en 267ms y host-bound — ROI bajo.
- **B=1 o B=3 builders**: B=2 óptimo medido (matriz K×B).

## 🎯 PENDIENTES (prioridad para 16 cores: reducir core-seconds de CPU o moverlos a GPU)

1. **`rounds_2to4` — EL blanco** (62-78% del thread-time de los provers, GPU 31% → host-bound). **PLAN COMPLETO: [impl-plan-rounds2to4.md](impl-plan-rounds2to4.md)** (2026-07-27, análisis de código con file:line, 7 pasos ordenados):
   - Boundary-zerofier inverses: batch-inverse de vectores LDE-size en CPU por constraint por tabla, subidos por PCIe → computarlos on-device (`batch_invert` GPU ya existe).
   - `decompose_and_extend_d2` (~5.5s thread-sum): identificar parte CPU vs GPU-wait con el nsys/phase_busy.
   - R3 OOD (~4s): barycentric GPU existe; ver qué cae a CPU y por qué.
   - R4 `deep_composition_poly_evals` (~5.7s): kernel deep existe; ¿D2H/host share?
   - R4 FRI commit (~2.9s) + queries (~1.7s).
   - ~~Cachear `LoweredCall` (lowering del constraint program) por tabla~~ → **HECHA 2026-07-27, medida −1.9%** (ver tabla de HECHAS, #14).
2. **LogUp aux decline**: fingerprint+invert ~0.85s thread siguen en CPU en algunas tablas. Primer paso barato: contador por causa de decline (threshold 2^10 / resident_aux_ok / descriptor).
3. ~~**`epoch_size_log2=21`**: medición barata nunca hecha~~ → **MEDIDO 2026-07-27 (10tx, binario tip+#14, 6 pares intercalados por duelo)**:
   - e20@K3 (default) 13.87s | e21@K2 **12.04s (−13.3%)** | e22@K1 **11.29s (−18.6%)** ← mejor config del box; gana 6/6 en ambos duelos y con menos varianza (sd 0.13s).
   - OOM VRAM (32GB): e21@K3 y e22@K2. RAM: e21 ~16GB/prove, e22 ~27GB/prove (el box de 60GB banca e22@K1).
   - Proofs e21 y e22 verifican, cross-binary con el tip ✅. Es config pura (`--epoch-size-log2` + `LAMBDA_VM_EPOCH_CONCURRENCY`), cero código.
   - **Con #14 + e22@K1: ~11.3s vs main 22.5s ≈ 2.0×.**
   - PENDIENTE de decisión (Joaquin): dónde encodearlo — defaults del CLI (riesgo: máquinas con menos RAM/VRAM) vs config de bench/deployment. OJO: medido a 10tx (2 epochs en e22 — pipeline degenerado); a 100tx (16 epochs @ e22) el tradeoff K↔tamaño puede cambiar — re-medir post-merge de #867.
4. **GPU-side (libera headroom, no wall directo)**: batch NTT levels (595ms en 3296 launches de 1 nivel; precedente `ntt_dit_8_levels_batched`), batch `keccak_merkle_level` (10k launches × 14µs), reducir launches totales (20.5k).
5. **Trace build interno**: `p2a_collect_cpu` 1.4s serial en el producer (16c) y `p4_bitwise_collect` ~1s/epoch en builders — paralelizar más fino o mover histogramas a GPU.
6. **Global prove CPU**: dilata con contención (2.7s@48c → 10.6s@32c); construir GM/L2G tables incrementalmente durante los epochs.
7. **D2H restante** (item 3 del baseline doc): DEEP result residente (consumer = FRI), R2 parts residency — estado parcial; re-evaluar con nsys fresco.
8. **Merkle CPU restante**: tablas bajo el threshold GPU (LDE <2^19) siguen commiteando en CPU.
9. **VRAM/robustez**: device-only aborta duro en OOM; los fallbacks `result.ok()` son silenciosos → agregar contadores/logs de fallback.
10. **Toolkit**: arreglar etapa `--nsys` en continuations (no genera .nsys-rep); phase_table.py con árboles por thread; unificar los dos backends NVTX (limpieza).

## 🐛 Deudas no-perf descubiertas
- Proof bytes NO deterministas entre corridas del mismo binario (preexistente; verifica OK) — afecta reproducibilidad/caching.
- Guest ELF viejo = trampa (ciclos distintos): SIEMPRE rebuildear el guest al benchear; main#859 (ecrecover) ya mejora ciclos del guest.
