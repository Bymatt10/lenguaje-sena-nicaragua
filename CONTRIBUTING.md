# Contribuir

¡Gracias por querer mejorar este proyecto! Hay dos formas de contribuir: **código/modelos** y **muestras de señas**.

## 1. Código y modelos

### Acuerdo de licencia (DCO)

Al abrir un Pull Request confirmas que tu contribución es tuya y la publicas bajo la **Apache License 2.0** (la misma del proyecto). Para que quede registro, añade al final del mensaje de tu PR una línea:

```
Signed-off-by: Tu Nombre <tu@email.com>
```

Aprende a configurarlo automáticamente con `git config user.name` y `git config user.email` y usando `git commit -s`.

### Estilo

- Python 3.10+ con type hints.
- Sin comentarios superfluos: el código debe explicarse solo.
- PRs pequeños y enfocados. Si tocas el modelo (`model.py`, `train_v2.py`), incluye los resultados LOSO antes y después.

### Cómo añadir una nueva seña al vocabulario

1. Graba muestras (`capture_samples.py`, 30+ por palabra).
2. Normaliza (`normalize_samples.py`) y crea los keypoints (`create_keypoints.py`).
3. Reentrena (`training_model.py` o `train_v2.py`) y mide LOSO (`evaluate_loso.py`).
4. Si el LOSO no cae respecto al baseline → push a `models/actions_<N>.keras` + `models/words.json`.
5. Ejecuta `auto_retrain.py` para actualizar `web/model/` y desplegar a la web.

## 2. Muestras desde la web (panel "Contribuir muestras")

Las muestras que envías son **datos personales** (keypoints de tu mano, pose y, opcionalmente, cara). Por favor:

- **No envíes muestras de otras personas sin su consentimiento.** Si filman niños o personas con discapacidad, pide autorización a su tutor/representante legal.
- **No incluyas información identificable** en el campo "Tu nombre o alias". Usa un alias (ej: `voluntario_lima_03`).
- La aplicación elimina los keypoints de la cara antes de enviarlos al servidor.
- Tus muestras se guardan en `dataset_contrib/` durante un máximo de **30 días** (configurable vía `LSN_RETENTION_DAYS`).

### Cómo borrar tus muestras

1. Pulsa el botón "Borrar mis datos" en la web (envía `DELETE /feedback/<token>`).
2. Si no tienes acceso a la web, envía un correo a `[EMAIL DEL MANTENEDOR]` con el alias que usaste y se eliminarán manualmente.

## 3. Privacidad y datos biométricos

Las muestras son **keypoints numéricos** (coordenadas x, y, z de la malla corporal y de las manos), no videos. Aun así pueden ser sensibles porque:

- Permiten reconstruir esqueletos 3D del voluntario.
- Si la cara está incluida, podrían usarse para reconocimiento facial.

Por eso:

- La web elimina los keypoints de la cara antes de subir.
- Las muestras en `dataset_contrib/` **no se publican** en el repo (están en `.gitignore`).
- Los videos originales (`dataset_videos/`) **no se publican** salvo decisión explícita del mantenedor.

## 4. Reportar un problema de seguridad

Por favor **no abras un issue público** si descubres una vulnerabilidad. Escribe a `[EMAIL DEL MANTENEDOR]` con:

- Descripción del problema.
- Pasos para reproducir.
- Impacto estimado.

Responderé en < 7 días. Una vez parcheado, publicaremos el aviso en `SECURITY.md` y te creditaremos (si quieres) en el aviso de la release.

## 5. Código de conducta

Se espera respeto en issues, PRs y discusiones. No se toleran insultos, acoso ni discriminación. El mantenedor puede cerrar cuentas o bloquear usuarios que incumplan.