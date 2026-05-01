# Predicción de cancelación de clientes — Interconnect

Proyecto final del *bootcamp* de Data Science de TripleTen. Trabajamos sobre
un caso de cancelación de clientes (*churn*) de Interconnect, operador de
telecomunicaciones, con la meta de dejar operativo un modelo predictivo que
ordene a los clientes por riesgo de cancelar para accionar campañas de
retención.

- **Fecha de corte de los datos:** 1 de febrero de 2020.
- **Cobertura:** 7 043 clientes únicos (5 174 activos + 1 869 cancelados).
- **Métrica principal:** AUC-ROC (umbral mínimo 0.75; objetivo ambicioso 0.88).
- **Resultado:** tras auditoría de nulos consolidada, validación cruzada,
  tuning de XGBoost, feature engineering y ensemble por promedio de los
  3 mejores, el ganador por métrica es el **ensemble top 3
  (XGBoost tuneado + CatBoost + RandomForest)** con
  **AUC 0.847 en test** y **0.859 en CV 5-fold**. Mejor modelo individual:
  **XGBoost tuneado (0.848 test)**. Supera el mínimo aceptable; el margen
  restante al objetivo 0.88 requiere fuentes de datos adicionales
  (consumo, soporte) no disponibles en esta fase.


## Qué contiene cada notebook

- **`Proyecto Final.ipynb`** — cuaderno de trabajo completo:
  - Carga + auditoría de calidad por dataset
  - **Auditoría de nulos consolidada post-merge** con razón probable por
    columna (ninguno es dato perdido real)
  - EDA: antigüedad, tipo de contrato, evolución por año
  - Cálculo de LTV + validación estadística (Mann-Whitney)
  - Tabla unificada para modelado con **feature engineering**
    (`AddonsCount`, `FiberNoSecurity`)
  - **Siete modelos base** (ya con los mejores hiperparámetros del tuning
    sincronizados) + cross-validation 5-fold estratificada
  - **Tabla comparativa con y sin CV** para confirmar estabilidad
  - **Optimización final**: `GridSearchCV` (DT / LogReg / RF) y
    `RandomizedSearchCV` (LGBM / XGBoost / CatBoost) con `verbose=2` y
    early stopping en boosters; MLP como `MLPClassifier` en un `Pipeline`
    con `StandardScaler` (CV 5-fold + early stopping nativo); ensemble
    promedio + stacking LogReg
  - **Tabla de importancia de variables** (consenso de 6 modelos)
  - **Función de exportación CSV/Excel para marketing** con 3 segmentos
    de riesgo y acción recomendada por segmento
  - Retrospectiva con los 5 retos principales enfrentados y sus soluciones
- **`Informe Mesa Directiva.ipynb`** — resumen ejecutivo con cifras
  clave, tabla con/sin CV, recomendación inmediata y siguientes pasos.
- **`Plan de Trabajo.ipynb`** — plan inicial, respuestas al líder de
  equipo y cierre de la segunda iteración con los resultados finales.

## Datos

Los cuatro CSV son archivos internos de Interconnect entregados por el
cliente. No se versionan por política de datos. Para correr el cuaderno es
necesario colocarlos manualmente en la carpeta `Data/`.

## Cómo correr el cuaderno principal

1. Activar el entorno virtual compartido:
   ```
   ..\..\.venv\Scripts\activate       # Windows
   source ../../.venv/bin/activate    # macOS / Linux
   ```
2. Instalar dependencias (solo primera vez):
   ```
   pip install -r requirements.txt
   ```
3. Para GPU en Windows (opcional), PyTorch con CUDA:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
4. Colocar los cuatro CSV en `Data/`.
5. Abrir `Notebook/Proyecto Final.ipynb` y ejecutar (Kernel → Restart &
   Run All). Tiempo total: ~3-4 minutos en GPU, ~7-10 min en CPU.

Al terminar se generan automáticamente `lista_retencion.csv` y
`lista_retencion.xlsx` con los 5 174 clientes activos segmentados por
riesgo y acción recomendada.

## Hallazgos principales

- **Pérdida acumulada por cancelación: ≈ 2.7 M USD** (15 % del LTV total).
- **Dos terceras partes de la fuga** vienen del segmento Month-to-month
  (tasa 28 % en 2014 → 55 % en 2019).
- **Los contratos anuales y bianuales** mantienen tasas bajas (13 % y 4 %).
- **Drivers del modelo (consenso de 6 modelos)**: tipo de contrato
  (Two year frena), antigüedad, cargos mensuales, Fiber optic (acelera),
  servicios de valor agregado (frenan).

Recomendación inmediata: **migrar clientes mensuales a contratos anuales o
bianuales**. La lista priorizada para marketing queda generada
automáticamente. **Nota operativa:** la lista actual se genera con
LightGBM tuneado (AUC 0.838 test / 0.850 CV). Regenerarla con el ensemble
top 3 para aprovechar el medio punto de AUC adicional en el ordenamiento
queda como mejora pendiente.

## Resultados del modelo (con y sin cross-validation)

| Modelo | AUC test | AUC CV (5-fold) |
|--------|---------:|----------------:|
| XGBoost tuneado | 0.848 | 0.857 |
| **Ensemble top 3 (XGBoost + CatBoost + RF)** | **0.847** | **0.859 ± 0.007** |
| LogisticRegression | 0.846 | 0.849 ± 0.010 |
| RandomForest | 0.845 | 0.852 ± 0.007 |
| LightGBM | 0.842 | 0.850 ± 0.006 |
| CatBoost | 0.836 | 0.855 ± 0.008 |
| XGBoost base | 0.828 | 0.842 ± 0.008 |
| DecisionTree | 0.805 | 0.827 ± 0.011 |
| MLP (sklearn) | 0.798 | 0.840 ± 0.009 |
| DummyClassifier | 0.500 | — |

La diferencia CV vs test es ≤ 0.04 en todos los casos → **los modelos son
estables**, no hay sobreajuste grave. El ensemble top-3 queda apenas
0.001 por debajo del mejor individual en test y lo iguala en CV; se
recomienda como modelo ganador por su mayor robustez.

## Tiempos registrados

Cada celda de modelado en §8 agrega a `resultados_modelos` las columnas
`tiempo_fit_s`, `tiempo_pred_s` y `tiempo_total_s`. Las celdas de tuning
añaden además `search_s` (CV + grid search) y `total_con_search_s`.

Optimizaciones aplicadas:

- **Early stopping** en el fit final de LightGBM / XGBoost / CatBoost,
  con un 15 % del train balanceado como validación estratificada (no se
  usa el test set).
- **MLP con `early_stopping=True` nativo de sklearn**, que reserva el
  10 % interno como validación y detiene cuando deja de mejorar.
- **`verbose`** activado donde aplica: `GridSearchCV` / `RandomizedSearchCV`
  con `verbose=2`, XGBoost `verbosity=1`, LightGBM `verbose=1`,
  CatBoost `verbose=100`.

## Retrospectiva — retos enfrentados

1. Desbalance de clases 73/27 — resuelto con **sobremuestreo** solo en train.
2. **Fuga por construcción en `DaysActive`** — escondía la fecha de
   cancelación y llevaba a AUC=1.0 ilegítimo. Se reemplazó por `Tenure`
   (antigüedad desde alta hasta fecha de corte). AUC resultante: 0.85.
3. Variables derivadas del target (`TotalCharges`, `LTV`) — excluidas.
4. Validación cruzada limpia: CV sobre datos originales con pesos de clase;
   el sobremuestreo se reserva para el fit final.
5. **Techo realista (~0.86)** — se aplicaron todas las palancas disponibles
   (tuning, ensemble, stacking, feature engineering). Cerrar la brecha al
   0.88 requiere fuentes nuevas (consumo, soporte, quejas).

Los mejores hiperparametros por modelo viven en un archivo dedicado:
`Hiperparametros_Modelos.md` en la raiz del proyecto.

## Autor

Fernando Muerza — TripleTen Data Science, Sprint 17 (proyecto final).
