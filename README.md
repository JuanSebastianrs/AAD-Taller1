# Taller 1 — Regresión Regularizada en Datos Genómicos

**Asignatura:** Análisis Avanzado de datos, por el profesor Andrés Nicolás López.
**Estudiantes:** Stefany Mojica, Sara Castillejo y Juan Sebastián Rodríguez.
*Maestría en Matemáticas Avanzadas y Ciencias de la Computación*
*Universidad del Rosario*

## Problema

El conjunto de datos `taller1.txt` contiene el perfil genómico de **1200 líneas celulares**
como modelos de cáncer. Se busca determinar cuáles de los **5000 genes** son relevantes
para predecir la efectividad del tratamiento anticáncer (variable continua).

**NOTA:** para correr correctamente este código, se debe agregar manualmente el archivo `taller1.txt` al clonar el repositorio. Dejamos aquí debajo un resumen solo con los resultados para una revisión rápida, pero en el archivo .py está el análisis completo.

---

## Punto 1 — Multicolinealidad

**¿Hay multicolinealidad en los datos?**

Sí. Las evidencias son:

| Indicador | Valor | Interpretación |
|---|---|---|
| Dimensionalidad | p=5000 >> n=1200 | X'X no es invertible |
| Rango de X | 1200 de 5000 | 3800 dependencias lineales exactas |
| Número de condición | 2.89 | >> 30, multicolinealidad severa |
| Componentes para 90% varianza | 918 de 5000 | Alta redundancia entre genes |

![Multicolinealidad](punto1_multicolinealidad.png)

---

## Punto 2 — Partición de datos 

| Conjunto | Observaciones |
|---|---|
| Entrenamiento | 1000 |
| Prueba | 200 |
| **Total** | **1200** |

- Semilla utilizada: `2026`

---

## Punto 3 — Selección de λ por validación cruzada

Método: **10-Fold Cross-Validation** sobre los 1000 datos de entrenamiento.

| Método | λ óptimo | ECM (CV) | Genes activos |
|---|---|---|---|
| Ridge | 52.140083 | 17.395507 | 5000 |
| Lasso | 0.070548 | 1.249969 | 117 |

![ECM vs Lambda](punto3_ridge_lasso_cv.png)

---

## Punto 4 — Ajuste con λ óptimos

Modelos ajustados sobre los **1000 datos de entrenamiento**.

| Métrica | Ridge | Lasso |
|---|---|---|
| λ óptimo | 52.140083 | 0.070548 |
| ECM entrenamiento | 0.002765 | 0.929672 |
| Genes activos | 5000 | 117 |

---

## Punto 5 — Selección del mejor modelo

Criterio: **ECM sobre los 200 datos de prueba** (uso único).

| Métrica | Ridge | Lasso |
|---|---|---|
| ECM prueba | 14.041272 | 1.153541 |
| Genes activos | 5000 | 117 |

**Modelo seleccionado: Lasso**
- Redujo el ECM en un **91.78%** respecto a Ridge.
- Utilizó solo el **2.34%** de los genes que usa Ridge.

---

## Punto 6 — Reajuste con los 1200 datos *(pendiente)*

> Sección por completar.

---

## Punto 7 — Trazas de coeficientes *(pendiente)*

> Sección por completar.

---

## Punto 8 — Conclusiones generales

> Sección por completar.

---

## Reproducibilidad
```bash
pip install -r requirements.txt
python taller1.py
```

**Semilla global:** `2026`
