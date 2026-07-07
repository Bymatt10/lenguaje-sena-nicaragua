# Estudio de ablación — Traductor LSN

**Fecha**: julio 2026 · **Benchmark**: leave-one-signer-out (LOSO) con 2 señantes
(elizabeth: 22 señas en video; prueba: 21 muestras del modo captura web).
**Protocolo**: cada experimento modifica UNA variable sobre el baseline; se entrena con un
señante y se evalúa con el otro (ambas direcciones); semillas fijas. Con solo 43 muestras
de prueba, se exige ≥ +4 puntos de mejora promedio para declarar una técnica ganadora.

## Resultados

| Experimento | Técnica | eval. elizabeth | eval. prueba | Promedio | Δ vs E0 |
|---|---|---|---|---|---|
| E0 | Baseline (v3: 702 features, 15 frames) | 36.4% | 42.9% | 39.6% | — |
| E1 | Rotación 3D de punto de vista (±25°) | 36.4% | 42.9% | 39.6% | +0.0 |
| E2a | Hand-dropout (p=0.15) | 36.4% | 33.3% | 34.8% | −4.8 |
| E2b | Decimación de fps (p=0.3, ×1.5–2.5) | 45.5% | 28.6% | 37.0% | −2.6 |
| E3 | Time-warping (2–3 nodos) | 31.8% | 28.6% | 30.2% | −9.4 |
| E4a | LSTM bidireccional | 27.3% | 33.3% | 30.3% | −9.3 |
| E4b | Label smoothing 0.1 | 27.3% | 33.3% | 30.3% | −9.3 |
| E5 | Ensemble ×3 semillas | 36.4% | 38.1% | 37.2% | −2.4 |
| **E6** | **Ventana temporal 15 → 25 frames** | **45.5%** | **42.9%** | **44.2%** | **+4.5** |
| E7 | MediaPipe model_complexity=2 | 18.2% | 19.0% | 18.6% | −21.0 |
| **COMBO** | **= E6 (única ganadora)** | **40.9%** | **47.6%** | **44.3%** | **+4.7** |

## Interpretación

- **E6 (25 frames) es la única mejora real**: las señas compuestas ("buenas noches" ≈ 4.5 s)
  perdían detalle al comprimirse a 15 frames. Adoptada en producción.
- **E7 parece catastrófico pero es un artefacto de protocolo**: los videos se re-extrajeron
  con `model_complexity=2` pero las muestras del modo captura web quedaron en calidad 1 —
  el modelo entrenó y evaluó con distribuciones de keypoints distintas. Conclusión: la
  calidad de MediaPipe debe ser **la misma en todas las fuentes**; no se adopta el cambio.
- **Las aumentaciones agresivas (E2a, E3) y los cambios de arquitectura (E4a, E4b)
  empeoran** con este volumen de datos: con 1 take por señante, distorsionar más las
  muestras o agrandar el modelo solo añade ruido/overfitting.
- **Lectura general**: con 2 señantes, las técnicas "gratis" ya tocaron techo (~44%).
  La curva de mejora ahora depende de agregar señantes al dataset (ver modo
  "Contribuir muestras" de la web).

## Reproducir

```bash
.venv/bin/python ablation.py        # corre los experimentos que falten (incremental)
cat ablation_results.json           # resultados crudos por fold
.venv/bin/python evaluate_loso.py   # solo el LOSO con la config actual
```
