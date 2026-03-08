import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")
np.random.seed(2026)

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
print("=" * 70)
print("CARGA DE DATOS")
print("=" * 70)

df = pd.read_csv("taller1.txt")
print(f"Dimensiones del dataset: {df.shape}")
print(f"  - Observaciones: {df.shape[0]}")
print(f"  - Variables:            {df.shape[1]}")
print(f"  - Genes (predictores):              {df.shape[1] - 1}")
print(f"\nPrimeras filas:")
print(df.iloc[:5, :8]) 

y = df["y"]
X = df.drop(columns=["y"])

print(f"\nVariable respuesta (y):")
print(y.describe())

# =============================================================================
# 2. PUNTO 1: ANÁLISIS DE MULTICOLINEALIDAD
# =============================================================================
print("\n" + "=" * 70)
print("PUNTO 1: ANÁLISIS DE MULTICOLINEALIDAD")
print("=" * 70)

n, p = X.shape
print(f"\n1a) Dimensionalidad:")
print(f"    n (observaciones) = {n}")
print(f"    p (predictores)   = {p}")
print(f"    Razón p/n          = {p/n:.2f}")
print(f"    → Como p ({p}) >> n ({n}), la matriz X'X de dimensión {p}x{p}")
print(f"      NO es invertible (rango máximo = {n}). Esto implica")
print(f"      multicolinealidad PERFECTA: existen infinitas combinaciones")
print(f"      lineales entre predictores que dan el vector cero.")

rango = np.linalg.matrix_rank(X.values)
print(f"\n1b) Rango de la matriz X:")
print(f"    Rango(X)   = {rango}")
print(f"    min(n, p)  = {min(n, p)}")
print(f"    → El rango es {rango}, confirmando que la matriz no tiene")
print(f"      rango completo en columnas ({p}).")

X_std = StandardScaler().fit_transform(X)

sv = np.linalg.svd(X_std, compute_uv=False)

sv_nonzero = sv[sv > 1e-10]
num_cond = sv_nonzero[0] / sv_nonzero[-1]
print(f"\n1c) Número de condición de X (estandarizada):")
print(f"    Valores singulares distintos de cero: {len(sv_nonzero)} de {len(sv)}")
print(f"    Valor singular máximo:  {sv_nonzero[0]:.4f}")
print(f"    Valor singular mínimo (>0): {sv_nonzero[-1]:.6f}")
print(f"    Número de condición:    {num_cond:.2f}")
if num_cond > 30:
    print(f"    → Número de condición >> 30, indica multicolinealidad SEVERA.")

print(f"\n1d) Análisis de correlaciones entre predictores:")

np.random.seed(2026)
sample_cols = np.random.choice(X.columns, size=min(500, p), replace=False)
corr_sample = X[sample_cols].corr()

mask_upper = np.triu(np.ones_like(corr_sample, dtype=bool), k=1)
corr_values = corr_sample.values[mask_upper]

print(f"    Estadísticas de |correlación| (muestra de {len(sample_cols)} genes):")
abs_corr = np.abs(corr_values)
print(f"    Media:   {abs_corr.mean():.4f}")
print(f"    Mediana: {np.median(abs_corr):.4f}")
print(f"    Max:     {abs_corr.max():.4f}")
print(f"    Pares con |r| > 0.8: {np.sum(abs_corr > 0.8)}")
print(f"    Pares con |r| > 0.9: {np.sum(abs_corr > 0.9)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(corr_values, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Correlación (r)", fontsize=12)
axes[0].set_ylabel("Frecuencia", fontsize=12)
axes[0].set_title("Distribución de correlaciones entre genes\n(muestra de 500 genes)",
                   fontsize=13)
axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)

explained_var = (sv_nonzero ** 2) / np.sum(sv_nonzero ** 2)
cumulative_var = np.cumsum(explained_var)
n_comp_90 = np.searchsorted(cumulative_var, 0.90) + 1
n_comp_95 = np.searchsorted(cumulative_var, 0.95) + 1

axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var * 100,
             color="steelblue", linewidth=2)
axes[1].axhline(y=90, color="red", linestyle="--", alpha=0.5, label=f"90% ({n_comp_90} comp.)")
axes[1].axhline(y=95, color="orange", linestyle="--", alpha=0.5, label=f"95% ({n_comp_95} comp.)")
axes[1].set_xlabel("Número de componentes", fontsize=12)
axes[1].set_ylabel("Varianza explicada acumulada (%)", fontsize=12)
axes[1].set_title("Varianza explicada acumulada (SVD)", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, min(200, len(cumulative_var)))

plt.tight_layout()
plt.savefig("punto1_multicolinealidad.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n    Componentes para explicar 90% de varianza: {n_comp_90} de {p}")
print(f"    Componentes para explicar 95% de varianza: {n_comp_95} de {p}")

# --- Conclusión Punto 1 ---
print(f"\n{'─' * 60}")
print("CONCLUSIÓN PUNTO 1:")
print("─" * 60)
print(f"""
SÍ hay multicolinealidad, Las razones son:

1. DIMENSIONALIDAD: p ({p}) >> n ({n}), por lo que la matriz X'X
   (de {p}x{p}) tiene rango como máximo {n}. Esto implica {p}-{rango}
   = {p - rango} dependencias lineales exactas entre los predictores.
   La multicolinealidad es PERFECTA en el sentido algebraico.

2. RANGO: El rango de X es {rango} < p = {p}, confirmando la
   singularidad de X'X.

3. REDUCCIÓN DIMENSIONAL: Solo {n_comp_90} de {p} componentes capturan
   el 90% de la varianza total, mostrando alta redundancia.

→ JUSTIFICACIÓN para usar métodos de regularización (Ridge/Lasso):
  Estos métodos resuelven el problema de la no invertibilidad de X'X
  agregando un término de penalización (λI para Ridge, λ|β| para Lasso).
""")

# =============================================================================
# 3. PUNTO 2: PARTICIÓN DE DATOS
# =============================================================================
print("=" * 70)
print("PUNTO 2: PARTICIÓN DE DATOS")
print("=" * 70)

SEED = 2026
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=1000,
    test_size=200,
    random_state=SEED
)

print(f"\nSemilla (random_state): {SEED}")
print(f"Conjunto de entrenamiento: {X_train.shape[0]} observaciones, {X_train.shape[1]} genes")
print(f"Conjunto de prueba:        {X_test.shape[0]} observaciones, {X_test.shape[1]} genes")
print(f"Total:                     {X_train.shape[0] + X_test.shape[0]} observaciones")

print(f"\nEstadísticas de y_train:")
print(f"  Media:    {y_train.mean():.4f}")
print(f"  Std:      {y_train.std():.4f}")
print(f"  Min:      {y_train.min():.4f}")
print(f"  Max:      {y_train.max():.4f}")

print(f"\nEstadísticas de y_test:")
print(f"  Media:    {y_test.mean():.4f}")
print(f"  Std:      {y_test.std():.4f}")
print(f"  Min:      {y_test.min():.4f}")
print(f"  Max:      {y_test.max():.4f}")

# =============================================================================
# 4. PUNTO 3: RIDGE Y LASSO CON VALIDACIÓN CRUZADA
# =============================================================================
print("\n" + "=" * 70)
print("PUNTO 3: REGRESIÓN RIDGE Y LASSO CON VALIDACIÓN CRUZADA")
print("=" * 70)
print("(Usando SOLO los 1000 datos de entrenamiento)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

alphas_ridge = np.logspace(-2, 6, 100)
alphas_lasso = np.logspace(-4, 2, 100)

K = 10
cv = KFold(n_splits=K, shuffle=True, random_state=SEED)

print(f"\nMétodo de validación externa: {K}-Fold Cross-Validation")
print(f"Semilla para CV:             {SEED}")
print(f"Métrica:                     ECM (Error Cuadrático Medio)")

# ─────────────────────────────────────────────────────────────────
# 4a. REGRESIÓN RIDGE
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("4a) REGRESIÓN RIDGE")
print("─" * 60)
print("    Calculando CV para cada λ... (esto puede tomar unos minutos)")

ridge_mse_per_alpha = np.zeros(len(alphas_ridge))
for i, alpha in enumerate(alphas_ridge):
    ridge_model = Ridge(alpha=alpha)
    scores = cross_val_score(ridge_model, X_train_scaled, y_train,
                             cv=cv, scoring="neg_mean_squared_error")
    ridge_mse_per_alpha[i] = -scores.mean()

ridge_best_idx = np.argmin(ridge_mse_per_alpha)
ridge_best_alpha = alphas_ridge[ridge_best_idx]
ridge_best_mse = ridge_mse_per_alpha[ridge_best_idx]

ridge_final = Ridge(alpha=ridge_best_alpha)
ridge_final.fit(X_train_scaled, y_train)

print(f"\n  Rango de λ explorado:  [{alphas_ridge[0]:.4f}, {alphas_ridge[-1]:.0f}]")
print(f"  Número de λ evaluados: {len(alphas_ridge)}")
print(f"  λ_r óptimo (Ridge):    {ridge_best_alpha:.6f}")
print(f"  ECM mínimo (CV):       {ridge_best_mse:.6f}")

n_nonzero_ridge = np.sum(np.abs(ridge_final.coef_) > 1e-8)
print(f"  Coeficientes ≠ 0:      {n_nonzero_ridge} de {p}")

# ─────────────────────────────────────────────────────────────────
# 4b. REGRESIÓN LASSO
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("4b) REGRESIÓN LASSO")
print("─" * 60)

lasso_cv = LassoCV(
    alphas=alphas_lasso,
    cv=cv,
    max_iter=10000,
    tol=1e-4,
    random_state=SEED,
    n_jobs=-1
)
lasso_cv.fit(X_train_scaled, y_train)

lasso_mse_per_alpha = lasso_cv.mse_path_.mean(axis=1) 
lasso_best_alpha = lasso_cv.alpha_
lasso_best_mse = lasso_mse_per_alpha[np.argmin(lasso_mse_per_alpha)]

print(f"\n  Rango de λ explorado:  [{alphas_lasso[0]:.6f}, {alphas_lasso[-1]:.2f}]")
print(f"  Número de λ evaluados: {len(alphas_lasso)}")
print(f"  λ_l óptimo (Lasso):    {lasso_best_alpha:.6f}")
print(f"  ECM mínimo (CV):       {lasso_best_mse:.6f}")

n_nonzero_lasso = np.sum(np.abs(lasso_cv.coef_) > 1e-8)
print(f"  Coeficientes ≠ 0:      {n_nonzero_lasso} de {p}")
print(f"  Coeficientes = 0:      {p - n_nonzero_lasso} de {p}")
print(f"  → Lasso selecciona {n_nonzero_lasso} genes relevantes.")

# ─────────────────────────────────────────────────────────────────
# 4c. GRÁFICOS: ECM vs λ
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Ridge: ECM vs λ ---
axes[0].plot(alphas_ridge, ridge_mse_per_alpha, color="steelblue", linewidth=2)
axes[0].axvline(x=ridge_best_alpha, color="red", linestyle="--", linewidth=1.5,
                label=f"λ_r óptimo = {ridge_best_alpha:.4f}")
axes[0].set_xscale("log")
axes[0].set_xlabel("λ (alpha)", fontsize=13)
axes[0].set_ylabel("ECM (Validación Cruzada)", fontsize=13)
axes[0].set_title(f"Ridge: ECM vs λ\nECM mínimo = {ridge_best_mse:.4f}", fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# --- Lasso: ECM vs λ ---
axes[1].plot(alphas_lasso, lasso_mse_per_alpha, color="darkorange", linewidth=2)
axes[1].axvline(x=lasso_best_alpha, color="red", linestyle="--", linewidth=1.5,
                label=f"λ_l óptimo = {lasso_best_alpha:.6f}")
axes[1].set_xscale("log")
axes[1].set_xlabel("λ (alpha)", fontsize=13)
axes[1].set_ylabel("ECM (Validación Cruzada)", fontsize=13)
axes[1].set_title(f"Lasso: ECM vs λ\nECM mínimo = {lasso_best_mse:.4f}", fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("punto3_ridge_lasso_cv.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────
# 4d. COMPARATIVO
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("RESUMEN COMPARATIVO")
print("─" * 60)
print(f"{'Método':<12} {'λ óptimo':>15} {'ECM (CV)':>12} {'Coef. ≠ 0':>12}")
print(f"{'─'*12} {'─'*15} {'─'*12} {'─'*12}")
print(f"{'Ridge':<12} {ridge_best_alpha:>15.6f} {ridge_best_mse:>12.6f} {n_nonzero_ridge:>12}")
print(f"{'Lasso':<12} {lasso_best_alpha:>15.6f} {lasso_best_mse:>12.6f} {n_nonzero_lasso:>12}")

print(f"\n{'─' * 60}")
print("CONCLUSIÓN PUNTO 3:")
print("─" * 60)
print(f"""
• Ridge (λ_r = {ridge_best_alpha:.6f}):
  - Retiene TODOS los {p} predictores (coeficientes pequeños pero ≠ 0).
  - ECM de validación cruzada: {ridge_best_mse:.6f}

• Lasso (λ_l = {lasso_best_alpha:.6f}):
  - Selecciona {n_nonzero_lasso} genes de {p} (reduce dimensionalidad).
  - ECM de validación cruzada: {lasso_best_mse:.6f}

• Método de validación externa utilizado: {K}-Fold Cross-Validation.
  Se eligió K={K} por ser el estándar para balancear sesgo-varianza
  en la estimación del error de predicción.
""")

# =============================================================================
# 5. PUNTO 4: AJUSTE FINAL CON LOS MEJORES λ (SOLO ENTRENAMIENTO)
# =============================================================================
print("\n" + "=" * 70)
print("PUNTO 4 (cont.): AJUSTE FINAL CON λ ÓPTIMOS")
print("=" * 70)
print("(Ajuste sobre los 1000 datos de entrenamiento)")

# ─────────────────────────────────────────────────────────────────
# 5a. MODELO FINAL RIDGE
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("5a) MODELO FINAL RIDGE")
print("─" * 60)

# ridge_final ya fue entrenado con ridge_best_alpha — no se re-entrena
y_pred_ridge_train = ridge_final.predict(X_train_scaled)
ecm_ridge_train    = mean_squared_error(y_train, y_pred_ridge_train)

ridge_coef_series = pd.Series(
    np.abs(ridge_final.coef_),
    index=[f"Gen_{i+1}" for i in range(p)]
)
top10_ridge = ridge_coef_series.nlargest(10)

print(f"  λ_r utilizado:         {ridge_best_alpha:.6f}")
print(f"  ECM en entrenamiento:  {ecm_ridge_train:.6f}")
print(f"  Coeficientes ≠ 0:      {n_nonzero_ridge} de {p}")
print(f"\n  Top 10 genes con mayor |coeficiente| en Ridge:")
print(f"  {'Gen':<12} {'|Coef|':>12}")
print(f"  {'─'*12} {'─'*12}")
for gen, coef in top10_ridge.items():
    print(f"  {gen:<12} {coef:>12.6f}")

# ─────────────────────────────────────────────────────────────────
# 5b. MODELO FINAL LASSO
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("5b) MODELO FINAL LASSO")
print("─" * 60)

# lasso_cv ya contiene el modelo ajustado — no se re-entrena
y_pred_lasso_train = lasso_cv.predict(X_train_scaled)
ecm_lasso_train    = mean_squared_error(y_train, y_pred_lasso_train)

lasso_coef_series = pd.Series(
    np.abs(lasso_cv.coef_),
    index=[f"Gen_{i+1}" for i in range(p)]
)
top10_lasso = lasso_coef_series[lasso_coef_series > 1e-8].nlargest(10)

print(f"  λ_l utilizado:         {lasso_best_alpha:.6f}")
print(f"  ECM en entrenamiento:  {ecm_lasso_train:.6f}")
print(f"  Genes seleccionados:   {n_nonzero_lasso} de {p}")
print(f"\n  Top 10 genes con mayor |coeficiente| en Lasso:")
print(f"  {'Gen':<12} {'|Coef|':>12}")
print(f"  {'─'*12} {'─'*12}")
for gen, coef in top10_lasso.items():
    print(f"  {gen:<12} {coef:>12.6f}")

# ─────────────────────────────────────────────────────────────────
# 5c. RESUMEN
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("RESUMEN PUNTO 4")
print("─" * 60)
print(f"{'Métrica':<25} {'Ridge':>12} {'Lasso':>12}")
print(f"{'─'*25} {'─'*12} {'─'*12}")
print(f"{'λ óptimo':<25} {ridge_best_alpha:>12.6f} {lasso_best_alpha:>12.6f}")
print(f"{'ECM entrenamiento':<25} {ecm_ridge_train:>12.6f} {ecm_lasso_train:>12.6f}")
print(f"{'Genes activos':<25} {n_nonzero_ridge:>12} {n_nonzero_lasso:>12}")

# =============================================================================
# 6. PUNTO 5: SELECCIÓN DEL MEJOR MODELO (ECM EN DATOS DE PRUEBA)
# =============================================================================
print("\n" + "=" * 70)
print("PUNTO 4: SELECCIÓN DEL MEJOR MODELO")
print("=" * 70)
print("(Evaluación sobre los 200 datos de prueba — uso único)")

# ─────────────────────────────────────────────────────────────────
# 6a. PREDICCIONES EN PRUEBA
# ─────────────────────────────────────────────────────────────────
y_pred_ridge_test = ridge_final.predict(X_test_scaled)
y_pred_lasso_test = lasso_cv.predict(X_test_scaled)

ecm_ridge_test = mean_squared_error(y_test, y_pred_ridge_test)
ecm_lasso_test = mean_squared_error(y_test, y_pred_lasso_test)

print(f"\n{'─' * 60}")
print("ECM EN DATOS DE PRUEBA")
print("─" * 60)
print(f"  ECM Ridge:  {ecm_ridge_test:.6f}")
print(f"  ECM Lasso:  {ecm_lasso_test:.6f}")

# ─────────────────────────────────────────────────────────────────
# 6b. SELECCIÓN
# ─────────────────────────────────────────────────────────────────
if ecm_ridge_test < ecm_lasso_test:
    mejor_modelo   = "Ridge"
    peor_modelo   = "Lasso"
    ecm_mejor      = ecm_ridge_test
    ecm_peor       = ecm_lasso_test
    genes_mejor     = n_nonzero_ridge
    genes_peor      = n_nonzero_lasso
    diferencia_gen  = n_nonzero_ridge / n_nonzero_lasso * 100
else:
    mejor_modelo   = "Lasso"
    peor_modelo   = "Ridge"
    ecm_mejor      = ecm_lasso_test
    ecm_peor       = ecm_ridge_test
    genes_mejor     = n_nonzero_lasso
    genes_peor      = n_nonzero_ridge
    diferencia_gen  = n_nonzero_lasso / n_nonzero_ridge  * 100

diferencia_pct = abs(ecm_ridge_test - ecm_lasso_test) / ecm_peor * 100

print(f"\n  → Modelo seleccionado: {mejor_modelo}")
print(f"  → Diferencia en ECM:   {diferencia_pct:.2f}%")
print(f"  → Diferencia en número de genes:   {diferencia_gen:.2f}%")

# ─────────────────────────────────────────────────────────────────
# 6c. RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("RESUMEN FINAL PUNTO 4")
print("─" * 60)
print(f"{'Métrica':<25} {'Ridge':>12} {'Lasso':>12}")
print(f"{'─'*25} {'─'*12} {'─'*12}")
print(f"{'λ óptimo':<25} {ridge_best_alpha:>12.6f} {lasso_best_alpha:>12.6f}")
print(f"{'ECM entrenamiento':<25} {ecm_ridge_train:>12.6f} {ecm_lasso_train:>12.6f}")
print(f"{'ECM prueba':<25} {ecm_ridge_test:>12.6f} {ecm_lasso_test:>12.6f}")
print(f"{'Genes activos':<25} {n_nonzero_ridge:>12} {n_nonzero_lasso:>12}")

print(f"""
CONCLUSIÓN PUNTO 4:
  • Modelo elegido: {mejor_modelo} (ECM = {ecm_mejor:.6f})

  • Sobre los 200 datos de prueba, {mejor_modelo} redujo un {diferencia_pct:.2f}% el ECM usando solo el {diferencia_gen:.2f}% de los genes que {peor_modelo}.

""")

# =============================================================================
# 7. punto 6: REAJUSTE CON LOS 1200 DATOS (PENDIENTE) 
# =============================================================================

# =============================================================================
# 8. punto 7: TRAZAS DE COEFICIENTES (PENDIENTE)
# =============================================================================




# =============================================================================
# GENERACIÓN AUTOMÁTICA DEL README 
# =============================================================================

readme_content = f"""# Taller 1 — Regresión Regularizada en Datos Genómicos

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
| Dimensionalidad | p={p} >> n={n} | X'X no es invertible |
| Rango de X | {rango} de {p} | {p - rango} dependencias lineales exactas |
| Número de condición | {num_cond:.2f} | >> 30, multicolinealidad severa |
| Componentes para 90% varianza | {n_comp_90} de {p} | Alta redundancia entre genes |

![Multicolinealidad](punto1_multicolinealidad.png)

---

## Punto 2 — Partición de datos 

| Conjunto | Observaciones |
|---|---|
| Entrenamiento | 1000 |
| Prueba | 200 |
| **Total** | **1200** |

- Semilla utilizada: `{SEED}`

---

## Punto 3 — Selección de λ por validación cruzada

Método: **{K}-Fold Cross-Validation** sobre los 1000 datos de entrenamiento.

| Método | λ óptimo | ECM (CV) | Genes activos |
|---|---|---|---|
| Ridge | {ridge_best_alpha:.6f} | {ridge_best_mse:.6f} | {n_nonzero_ridge} |
| Lasso | {lasso_best_alpha:.6f} | {lasso_best_mse:.6f} | {n_nonzero_lasso} |

![ECM vs Lambda](punto3_ridge_lasso_cv.png)

---

## Punto 4 — Ajuste con λ óptimos

Modelos ajustados sobre los **1000 datos de entrenamiento**.

| Métrica | Ridge | Lasso |
|---|---|---|
| λ óptimo | {ridge_best_alpha:.6f} | {lasso_best_alpha:.6f} |
| ECM entrenamiento | {ecm_ridge_train:.6f} | {ecm_lasso_train:.6f} |
| Genes activos | {n_nonzero_ridge} | {n_nonzero_lasso} |

---

## Punto 5 — Selección del mejor modelo

Criterio: **ECM sobre los 200 datos de prueba** (uso único).

| Métrica | Ridge | Lasso |
|---|---|---|
| ECM prueba | {ecm_ridge_test:.6f} | {ecm_lasso_test:.6f} |
| Genes activos | {n_nonzero_ridge} | {n_nonzero_lasso} |

**Modelo seleccionado: {mejor_modelo}**
- Redujo el ECM en un **{diferencia_pct:.2f}%** respecto a {peor_modelo}.
- Utilizó solo el **{diferencia_gen:.2f}%** de los genes que usa {peor_modelo}.

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

**Semilla global:** `{SEED}`
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("\n" + "=" * 70)
print("README.md generado exitosamente")
print("=" * 70)