'''
Análisis de RAW_interactions (user_id, recipe_id, date, rating, review).

Se usa desde dataset_analises_main.py (opción "interactions"). Cada bloque se apoya
en lo visto en las prácticas:

    1. Descriptivo      -> frequency / value_range (rating, usuarios, recetas, fechas, reviews)
    2. Recomendador     -> sparsity, cold start, coherencia con RAW_recipes, duplicados
    3. Modelos          -> Lab 2 (métricas, ROC, split sin fuga), P4 (LogisticRegression, C,
                           GridSearchCV), Lab 9 (TF-IDF + MultinomialNB)
    4. Clustering       -> P7 (StandardScaler, K-means, codo y silhouette)

Funciones:
    basic_overview(df)
        Dimensiones, nulos, duplicados y rango de fechas.
    rating_distribution(df)
        Frecuencia de cada rating (0-5) con gráfico.
    activity_distribution(df, key)
        Nº de interacciones por user_id o recipe_id: estadísticas, top y cola larga.
    temporal_distribution(df)
        Interacciones por año.
    review_length_stats(df)
        Longitud de la review en palabras.
    sparsity(df)
        Densidad/sparsity de la matriz usuario x receta.
    cold_start(df)
        % de usuarios y recetas con pocas interacciones.
    coherence(df_interactions, df_recipes)
        recipe_id sin receta y recetas sin interacciones.
    baseline_models(df)
        Clasifica rating alto/bajo desde la review: Dummy vs Naive Bayes vs Logística.
    cluster_users(df)
        K-means sobre el perfil de cada usuario (codo + silhouette).
    analise_interactions(df_interactions, df_recipes, part)
        Lanza uno o todos los bloques.
'''

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Carpeta donde está este archivo (Dataset_analise/)
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "Resultats" / "interactions"
GRAPH_DIR = BASE_DIR / "grafos" / "interactions"

PARTS = ["all", "descriptive", "recommender", "models", "clusters"]
RANDOM_STATE = 0


# --------------------------------------------------------------------------- utilidades

def _save_csv(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Results saved to {path}")


def _save_fig(fig, name):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    path = GRAPH_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to {path}")


def _describe(values, name):
    '''Mismas estadísticas que get_stats (count, mean, std, min, 25/50/75%, max) + p95 y p99.'''
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    return {
        "feature": name,
        "count": values.size,
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "25%": np.percentile(values, 25),
        "50%": np.percentile(values, 50),
        "75%": np.percentile(values, 75),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
        "max": values.max(),
    }


# --------------------------------------------------------------------------- 1. descriptivo

def basic_overview(df):
    '''Dimensiones, nulos, duplicados (user_id, recipe_id) y rango de fechas.'''
    dates = pd.to_datetime(df["date"], errors="coerce")
    rows = {
        "filas": len(df),
        "usuarios_unicos": df["user_id"].nunique(),
        "recetas_unicas": df["recipe_id"].nunique(),
        "duplicados_user_recipe": int(df.duplicated(["user_id", "recipe_id"]).sum()),
        "duplicados_fila_completa": int(df.duplicated().sum()),
        "fecha_min": dates.min(),
        "fecha_max": dates.max(),
    }
    for col in df.columns:
        rows[f"nulos_{col}"] = int(df[col].isna().sum())

    overview = pd.DataFrame({"metric": list(rows), "value": list(rows.values())})
    print(overview.to_string(index=False))
    _save_csv(overview, "overview")
    return overview


def rating_distribution(df):
    '''Frecuencia de cada rating. El 0 en Food.com significa "sin puntuación, solo reseña".'''
    counts = df["rating"].value_counts().sort_index()
    table = pd.DataFrame({
        "rating": counts.index,
        "freq": counts.values,
        "percent": (counts.values / counts.sum() * 100).round(3),
    })
    print(table.to_string(index=False))
    _save_csv(table, "rating_frequency")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(table["rating"].astype(str), table["freq"], color="#4C72B0")
    ax.bar_label(bars, fontsize=8)
    ax.set_title("Distribución del rating")
    ax.set_xlabel("rating")
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, "rating_frequency")
    return table


def activity_distribution(df, key="user_id", top_n=20):
    '''
    Nº de interacciones por usuario o por receta. Muestra la cola larga típica de un
    recomendador: pocos elementos concentran muchas interacciones.
    '''
    if key not in ("user_id", "recipe_id"):
        print(f"Clave '{key}' no reconocida. Opciones: user_id, recipe_id")
        return None

    counts = df[key].value_counts()
    stats = pd.DataFrame([_describe(counts.values, f"interactions_per_{key}")])
    print(stats.to_string(index=False))
    _save_csv(stats, f"interactions_per_{key}_stats")

    top = counts.head(top_n).rename_axis(key).reset_index(name="interactions")
    _save_csv(top, f"interactions_per_{key}_top")

    # Frecuencia de frecuencias: cuántos elementos tienen exactamente n interacciones
    freq_of_freq = counts.value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(counts.values, bins=50, color="#4C72B0")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"interacciones por {key}")
    ax1.set_ylabel("nº de elementos (log)")
    ax1.set_title("Histograma")
    ax2.loglog(freq_of_freq.index, freq_of_freq.values, ".", color="#C44E52")
    ax2.set_xlabel("nº de interacciones (log)")
    ax2.set_ylabel("nº de elementos (log)")
    ax2.set_title("Cola larga")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Actividad por {key}")
    _save_fig(fig, f"interactions_per_{key}")
    return counts


def temporal_distribution(df):
    '''Interacciones por año.'''
    years = pd.to_datetime(df["date"], errors="coerce").dt.year.dropna().astype(int)
    counts = years.value_counts().sort_index()
    table = pd.DataFrame({"year": counts.index, "interactions": counts.values})
    print(table.to_string(index=False))
    _save_csv(table, "interactions_per_year")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["year"].astype(str), table["interactions"], color="#4C72B0")
    ax.set_title("Interacciones por año")
    ax.set_xlabel("año")
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save_fig(fig, "interactions_per_year")
    return table


def review_length_stats(df):
    '''Longitud de la review en palabras (mismo criterio que get_labels_and_values_string).'''
    lengths = df["review"].fillna("").astype(str).str.split().str.len()
    stats = pd.DataFrame([_describe(lengths.values, "review_n_words")])
    print(stats.to_string(index=False))
    _save_csv(stats, "review_length_stats")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths.clip(upper=lengths.quantile(0.99)), bins=50, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_title("Longitud de la review (recortada en p99)")
    ax.set_xlabel("palabras")
    ax.set_ylabel("Frequency (log)")
    ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, "review_length")
    return stats


# --------------------------------------------------------------------------- 2. recomendador

def sparsity(df):
    '''Sparsity = 1 - interacciones / (usuarios x recetas).'''
    n_users = df["user_id"].nunique()
    n_recipes = df["recipe_id"].nunique()
    n_inter = len(df.drop_duplicates(["user_id", "recipe_id"]))
    density = n_inter / (n_users * n_recipes)

    table = pd.DataFrame({
        "metric": ["usuarios", "recetas", "pares_user_recipe", "celdas_matriz",
                   "densidad", "sparsity"],
        "value": [n_users, n_recipes, n_inter, n_users * n_recipes, density, 1 - density],
    })
    print(table.to_string(index=False))
    _save_csv(table, "sparsity")
    return table


def cold_start(df, thresholds=(1, 2, 5, 10)):
    '''% de usuarios y recetas con como máximo t interacciones. Sirve para elegir un filtro mínimo.'''
    per_user = df["user_id"].value_counts()
    per_recipe = df["recipe_id"].value_counts()

    rows = []
    for t in thresholds:
        rows.append({
            "max_interacciones": t,
            "usuarios": int((per_user <= t).sum()),
            "usuarios_pct": round((per_user <= t).mean() * 100, 2),
            "recetas": int((per_recipe <= t).sum()),
            "recetas_pct": round((per_recipe <= t).mean() * 100, 2),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    _save_csv(table, "cold_start")
    return table


def coherence(df_interactions, df_recipes):
    '''recipe_id de interactions que no están en recipes (id) y recetas sin ninguna interacción.'''
    inter_ids = set(df_interactions["recipe_id"].unique())
    recipe_ids = set(df_recipes["id"].unique())

    table = pd.DataFrame({
        "metric": ["recetas_en_recipes", "recetas_en_interactions",
                   "interactions_sin_receta", "recetas_sin_interacciones"],
        "value": [len(recipe_ids), len(inter_ids),
                  len(inter_ids - recipe_ids), len(recipe_ids - inter_ids)],
    })
    print(table.to_string(index=False))
    _save_csv(table, "coherence")
    return table


# --------------------------------------------------------------------------- 3. modelos

def _metrics(name, y_true, y_pred, y_score):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
    }


def baseline_models(df, threshold=4, sample_size=50000, max_features=20000):
    '''
    Predice si un rating es alto (rating >= threshold) a partir del texto de la review.

    - Se descartan los rating 0 (no son una puntuación).
    - Split estratificado; el TF-IDF va dentro del Pipeline para ajustarse solo con train
      (la regla del Lab 2: nunca ajustar el scaler/vectorizador con el test).
    - Se compara con un clasificador trivial: con clases desbalanceadas el accuracy engaña.
    - Logística: C elegido con GridSearchCV (P4). Naive Bayes: TF-IDF + MultinomialNB (Lab 9).
    '''
    data = df[df["rating"] > 0].copy()
    data["review"] = data["review"].fillna("").astype(str)
    data = data[data["review"].str.strip() != ""]

    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=RANDOM_STATE)

    y = (data["rating"] >= threshold).astype(int)
    X = data["review"]

    balance = y.value_counts(normalize=True).sort_index() * 100
    print(f"Muestra: {len(data)} reviews | clase 0 (bajo): {balance.get(0, 0):.1f}% "
          f"| clase 1 (alto, rating >= {threshold}): {balance.get(1, 0):.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    def tfidf():
        return TfidfVectorizer(max_features=max_features, stop_words="english", min_df=2)

    models = {
        "Dummy (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes (TF-IDF)": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB())]),
        "Logística (TF-IDF)": GridSearchCV(
            Pipeline([("tfidf", tfidf()),
                      ("clf", LogisticRegression(max_iter=1000))]),
            param_grid={"clf__C": [0.1, 1, 10]},
            scoring="f1", cv=3, n_jobs=-1,
        ),
    }

    results, scores, preds = [], {}, {}
    for name, model in models.items():
        if name.startswith("Dummy"):
            model.fit(X_train.to_numpy().reshape(-1, 1), y_train)
            y_pred = model.predict(X_test.to_numpy().reshape(-1, 1))
            y_score = model.predict_proba(X_test.to_numpy().reshape(-1, 1))[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
        if isinstance(model, GridSearchCV):
            print(f"Mejor C (GridSearchCV): {model.best_params_['clf__C']}")
        results.append(_metrics(name, y_test, y_pred, y_score))
        scores[name], preds[name] = y_score, y_pred

    table = pd.DataFrame(results).round(4)
    print(table.to_string(index=False))
    _save_csv(table, "baseline_models")

    # ROC de todos los modelos en una misma gráfica (ejercicio del Lab 2)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_test, y_score):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC: rating >= {threshold} desde la review")
    ax.legend(loc="lower right", fontsize=8)
    _save_fig(fig, "baseline_roc")

    # Matriz de confusión del mejor modelo por F1
    best = table.sort_values("f1", ascending=False).iloc[0]["model"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds[best], display_labels=["Bajo", "Alto"], ax=ax
    )
    ax.set_title(f"Matriz de confusión: {best}")
    _save_fig(fig, "baseline_confusion")
    return table


# --------------------------------------------------------------------------- 4. clustering

def cluster_users(df, k_range=range(2, 9), min_interactions=2, silhouette_sample=10000):
    '''
    K-means sobre el perfil de cada usuario: nº de interacciones (log), rating medio,
    desviación del rating y longitud media de la review. Se estandariza antes (P7) y se
    miran el codo y el silhouette para escoger K.
    '''
    rated = df[df["rating"] > 0].copy()
    rated["n_words"] = rated["review"].fillna("").astype(str).str.split().str.len()

    profile = rated.groupby("user_id").agg(
        n_interactions=("rating", "size"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        mean_words=("n_words", "mean"),
    )
    profile["std_rating"] = profile["std_rating"].fillna(0)
    profile = profile[profile["n_interactions"] >= min_interactions]

    if len(profile) <= max(k_range):
        print("Hay muy pocos usuarios para hacer clustering con esos filtros.")
        return None

    features = profile.copy()
    features["n_interactions"] = np.log1p(features["n_interactions"])  # cola larga
    X = StandardScaler().fit_transform(features)

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(
            X, km.labels_, sample_size=min(silhouette_sample, len(X)),
            random_state=RANDOM_STATE
        )
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

    table = pd.DataFrame(rows).round(4)
    print(f"Usuarios con >= {min_interactions} valoraciones: {len(profile)}")
    print(table.to_string(index=False))
    _save_csv(table, "cluster_users_k")

    best_k = int(table.loc[table["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(X)
    profile["cluster"] = km.labels_
    summary = profile.groupby("cluster").agg(
        usuarios=("n_interactions", "size"),
        n_interactions=("n_interactions", "mean"),
        mean_rating=("mean_rating", "mean"),
        std_rating=("std_rating", "mean"),
        mean_words=("mean_words", "mean"),
    ).round(3).reset_index()
    print(f"K elegido por silhouette: {best_k}")
    print(summary.to_string(index=False))
    _save_csv(summary, "cluster_users_summary")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(table["k"], table["inertia"], "o-", color="#4C72B0")
    ax1.set_title("Método del codo (inercia)")
    ax2.plot(table["k"], table["silhouette"], "o-", color="#C44E52")
    ax2.axvline(best_k, color="gray", ls="--")
    ax2.set_title("Silhouette (más alto = mejor)")
    for ax in (ax1, ax2):
        ax.set_xlabel("K")
        ax.spines[["top", "right"]].set_visible(False)
    _save_fig(fig, "cluster_users_k")
    return summary


# --------------------------------------------------------------------------- punto de entrada

def analise_interactions(df_interactions, df_recipes=None, part="all"):
    '''
    Lanza el análisis de RAW_interactions.

    Parámetros:
        df_interactions (pd.DataFrame): RAW_interactions.
        df_recipes (pd.DataFrame | None): RAW_recipes (solo para la coherencia entre datasets).
        part (str): "all", "descriptive", "recommender", "models" o "clusters".
    '''
    if part not in PARTS:
        print(f"Bloque '{part}' no reconocido. Opciones: {PARTS}")
        return

    if part in ("all", "descriptive"):
        print("\n=== Descriptivo ===")
        basic_overview(df_interactions)
        rating_distribution(df_interactions)
        activity_distribution(df_interactions, "user_id")
        activity_distribution(df_interactions, "recipe_id")
        temporal_distribution(df_interactions)
        review_length_stats(df_interactions)

    if part in ("all", "recommender"):
        print("\n=== Recomendador ===")
        sparsity(df_interactions)
        cold_start(df_interactions)
        if df_recipes is not None:
            coherence(df_interactions, df_recipes)

    if part in ("all", "models"):
        print("\n=== Modelos base (rating alto/bajo desde la review) ===")
        baseline_models(df_interactions)

    if part in ("all", "clusters"):
        print("\n=== Clustering de usuarios ===")
        cluster_users(df_interactions)