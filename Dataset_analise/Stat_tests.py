'''
Tests estadísticos sobre RAW_interactions + RAW_recipes.

Se usa desde dataset_analises_main.py (opción "stat_tests"). Responde a preguntas del tipo
"¿depende el rating de ...?" con tests de hipótesis, y los complementa con el tamaño de efecto.

Con ~1 millón de filas casi todo sale "significativo" (p ~ 0), así que el p-valor por sí solo
no sirve: se reporta siempre V de Cramér (chi-cuadrado) o d de Cohen (t-test).

Funciones:
    prepare_merged(df_interactions, df_recipes)
        Une interactions con recipes por recipe_id y deja solo los rating 1-5.
    chi2_rating_vs_bins(merged, feature, n_bins)
        Chi-cuadrado de independencia: rating x tramos (cuantiles) de una variable numérica.
    chi2_rating_vs_tags(merged, df_recipes, tags, n_tags)
        Chi-cuadrado de independencia: rating x (tiene / no tiene el tag), para varios tags.
    ttest_fast_vs_slow(merged, unit)
        Welch + Mann-Whitney: rating de recetas rápidas vs lentas (mediana de minutes).
    ttest_tag(merged, df_recipes, tags, n_tags, unit)
        Welch + Mann-Whitney: rating de recetas con vs sin un tag.
    analise_tests(df_interactions, df_recipes, part)
        Lanza uno o todos los bloques.
'''

import ast
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Carpeta donde está este archivo (Dataset_analise/)
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "Resultats" / "tests"
GRAPH_DIR = BASE_DIR / "grafos" / "tests"

PARTS = ["all", "chi2", "ttest"]
NUMERIC_FEATURES = ["minutes", "n_steps", "n_ingredients"]


# --------------------------------------------------------------------------- utilidades

def _save_csv(df, name, index=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = OUT_DIR / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8")
    print(f"Results saved to {path}")


def _save_fig(fig, name):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    path = GRAPH_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to {path}")


def cramers_v(chi2, n, table_shape):
    '''V de Cramér: tamaño de efecto del chi-cuadrado, entre 0 (independientes) y 1.'''
    k = min(table_shape[0] - 1, table_shape[1] - 1)
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else np.nan


def interpret_cramers_v(v):
    if v < 0.1:
        return "despreciable"
    if v < 0.3:
        return "pequeño"
    if v < 0.5:
        return "moderado"
    return "grande"


def cohens_d(a, b):
    '''d de Cohen con desviación típica combinada.'''
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2:
        return "despreciable"
    if d < 0.5:
        return "pequeño"
    if d < 0.8:
        return "mediano"
    return "grande"


def _parse_list(cell):
    try:
        return ast.literal_eval(cell) if pd.notna(cell) else []
    except (ValueError, SyntaxError):
        return []


# --------------------------------------------------------------------------- preparación

def prepare_merged(df_interactions, df_recipes):
    '''
    Une interactions con las variables de recipes que usan los tests.

    - Se descartan los rating 0 (en Food.com significan "sin puntuación").
    - Se hace un inner join: las interacciones sin receta se pierden (ver coherence()
      en Interactions.py para saber cuántas son).
    '''
    recipes = df_recipes[["id", "minutes", "n_steps", "n_ingredients"]].rename(
        columns={"id": "recipe_id"}
    )
    rated = df_interactions[df_interactions["rating"] > 0][["user_id", "recipe_id", "rating"]]
    merged = rated.merge(recipes, on="recipe_id", how="inner")

    print(f"Interacciones con rating: {len(rated)} | tras el merge con recipes: {len(merged)}")
    return merged


def _by_recipe(merged):
    '''Una fila por receta (rating medio). Evita contar muchas veces la misma receta.'''
    return merged.groupby("recipe_id").agg(
        rating=("rating", "mean"),
        n_ratings=("rating", "size"),
        minutes=("minutes", "first"),
        n_steps=("n_steps", "first"),
        n_ingredients=("n_ingredients", "first"),
    ).reset_index()


def _tags_series(df_recipes):
    '''Listas de tags ya parseadas (columna "tags_list" si existe; si no, se parsean).'''
    if "tags_list" in df_recipes.columns:
        return df_recipes["tags_list"]
    return df_recipes["tags"].apply(_parse_list)


def _tag_flags(df_recipes, tag):
    '''Serie recipe_id -> True/False según si la receta tiene el tag.'''
    tags_list = _tags_series(df_recipes)
    return pd.Series(tags_list.apply(lambda lst: tag in lst).to_numpy(),
                     index=df_recipes["id"].to_numpy())


def top_tags(df_recipes, n_tags=10, min_prev=0.05, max_prev=0.5):
    '''
    Tags más frecuentes cuya presencia no sea casi universal ni casi inexistente.
    Un tag que está en el 99 % de las recetas no permite comparar "con vs sin".
    '''
    tags_list = _tags_series(df_recipes)
    counts = tags_list.explode().value_counts()
    prevalence = counts / len(df_recipes)
    selected = prevalence[(prevalence >= min_prev) & (prevalence <= max_prev)]
    return list(selected.head(n_tags).index)


# --------------------------------------------------------------------------- chi-cuadrado

def _chi2_from_table(table, label):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())
    v = cramers_v(chi2, n, table.shape)
    return {
        "test": label,
        "n": n,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "cramers_v": v,
        "efecto": interpret_cramers_v(v),
        "min_esperado": float(expected.min()),  # el test es fiable si es >= 5
    }


def _heatmap(table, title, name):
    '''Porcentaje de cada rating dentro de cada grupo (cada fila suma 100).'''
    pct = table.div(table.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(pct) + 2))
    im = ax.imshow(pct.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(pct.shape[1]))
    ax.set_xticklabels(pct.columns)
    ax.set_yticks(range(pct.shape[0]))
    ax.set_yticklabels([str(i) for i in pct.index], fontsize=8)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            ax.text(j, i, f"{pct.iat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("rating")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="% dentro del grupo")
    _save_fig(fig, name)


def chi2_rating_vs_bins(merged, feature, n_bins=4):
    '''
    H0: el rating es independiente del tramo de `feature`.

    Los tramos son cuantiles (qcut), así que los outliers de minutes no distorsionan nada.
    '''
    if feature not in NUMERIC_FEATURES:
        print(f"Feature '{feature}' no reconocida. Opciones: {NUMERIC_FEATURES}")
        return None

    bins = pd.qcut(merged[feature], q=n_bins, duplicates="drop")
    table = pd.crosstab(bins, merged["rating"])
    table.index = table.index.astype(str)

    result = _chi2_from_table(table, f"rating x {feature} ({table.shape[0]} tramos)")
    _save_csv(table, f"chi2_contingency_{feature}", index=True)
    _heatmap(table, f"% de cada rating por tramo de {feature}", f"chi2_heatmap_{feature}")
    return result


def chi2_rating_vs_tags(merged, df_recipes, tags=None, n_tags=10):
    '''
    H0: el rating es independiente de que la receta tenga el tag.

    Se hacen varios tests a la vez, así que se añade la corrección de Bonferroni
    (p_bonferroni = p * nº de tests): con muchos tests, alguno saldría significativo por azar.
    '''
    tags = tags or top_tags(df_recipes, n_tags)
    rows = []
    for tag in tags:
        has_tag = merged["recipe_id"].map(_tag_flags(df_recipes, tag)).fillna(False).astype(bool)
        table = pd.crosstab(has_tag, merged["rating"])
        if table.shape[0] < 2:
            print(f"Aviso: el tag '{tag}' no tiene grupo con/sin tag; se ignora.")
            continue
        rows.append(_chi2_from_table(table, f"rating x tag '{tag}'"))

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bonferroni"] = (result["p_value"] * len(result)).clip(upper=1.0)
    return result


# --------------------------------------------------------------------------- t-test

def _compare_groups(group_a, group_b, label, name_a, name_b, unit):
    '''Welch (no asume varianzas iguales) + Mann-Whitney (no asume normalidad) + d de Cohen.'''
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    t_stat, p_t = stats.ttest_ind(a, b, equal_var=False)
    u_stat, p_u = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    d = cohens_d(a, b)

    return {
        "test": label,
        "unidad": unit,
        "grupo_A": name_a,
        "grupo_B": name_b,
        "n_A": len(a),
        "n_B": len(b),
        "media_A": a.mean(),
        "media_B": b.mean(),
        "diferencia": a.mean() - b.mean(),
        "t_welch": t_stat,
        "p_welch": p_t,
        "p_mannwhitney": p_u,
        "cohens_d": d,
        "efecto": interpret_cohens_d(d),
    }


def _data_for_unit(merged, unit):
    if unit == "recipe":
        return _by_recipe(merged)
    if unit == "interaction":
        return merged
    raise ValueError("unit debe ser 'interaction' o 'recipe'")


def ttest_fast_vs_slow(merged, unit="interaction"):
    '''
    H0: el rating medio de las recetas rápidas (minutes <= mediana) es igual al de las lentas.

    unit="interaction": una fila por valoración. Las recetas muy valoradas pesan más y las
                        valoraciones de una misma receta no son independientes.
    unit="recipe":      una fila por receta (rating medio). Es más conservador y más defendible.
    '''
    data = _data_for_unit(merged, unit)
    median = data["minutes"].median()
    fast = data.loc[data["minutes"] <= median, "rating"]
    slow = data.loc[data["minutes"] > median, "rating"]
    return _compare_groups(fast, slow, f"rating: rápidas vs lentas (mediana = {median:g} min)",
                           "rápidas", "lentas", unit)


def ttest_tag(merged, df_recipes, tags=None, n_tags=10, unit="interaction"):
    '''H0: el rating medio de las recetas con el tag es igual al de las que no lo tienen.'''
    tags = tags or top_tags(df_recipes, n_tags)
    data = _data_for_unit(merged, unit)

    rows = []
    for tag in tags:
        has_tag = data["recipe_id"].map(_tag_flags(df_recipes, tag)).fillna(False).astype(bool)
        with_tag, without_tag = data.loc[has_tag, "rating"], data.loc[~has_tag, "rating"]
        if len(with_tag) < 2 or len(without_tag) < 2:
            print(f"Aviso: el tag '{tag}' no tiene suficientes datos; se ignora.")
            continue
        rows.append(_compare_groups(with_tag, without_tag, f"rating: con vs sin tag '{tag}'",
                                    "con tag", "sin tag", unit))

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_bonferroni"] = (result["p_welch"] * len(result)).clip(upper=1.0)
    return result


# --------------------------------------------------------------------------- punto de entrada

def analise_tests(df_interactions, df_recipes, part="all"):
    '''
    Lanza los tests estadísticos.

    Parámetros:
        df_interactions (pd.DataFrame): RAW_interactions.
        df_recipes (pd.DataFrame): RAW_recipes (obligatorio: los tests cruzan ambos datasets).
        part (str): "all", "chi2" o "ttest".
    '''
    if part not in PARTS:
        print(f"Bloque '{part}' no reconocido. Opciones: {PARTS}")
        return
    if df_recipes is None:
        print("Los tests necesitan RAW_recipes.")
        return

    # Los tags vienen como texto "['a', 'b']": se parsean una sola vez para todos los tests
    df_recipes = df_recipes.assign(tags_list=df_recipes["tags"].apply(_parse_list))

    merged = prepare_merged(df_interactions, df_recipes)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    if part in ("all", "chi2"):
        print("\n=== Chi-cuadrado de independencia ===")
        rows = [chi2_rating_vs_bins(merged, feature) for feature in NUMERIC_FEATURES]
        bins_result = pd.DataFrame([r for r in rows if r is not None])
        tags_result = chi2_rating_vs_tags(merged, df_recipes)
        chi2_result = pd.concat([bins_result, tags_result], ignore_index=True)
        print(chi2_result.round(4).to_string(index=False))
        _save_csv(chi2_result, "chi2_results")

    if part in ("all", "ttest"):
        print("\n=== T-test de Welch + Mann-Whitney ===")
        rows = []
        for unit in ("interaction", "recipe"):
            rows.append(pd.DataFrame([ttest_fast_vs_slow(merged, unit)]))
            rows.append(ttest_tag(merged, df_recipes, unit=unit))
        ttest_result = pd.concat(rows, ignore_index=True)
        print(ttest_result.round(4).to_string(index=False))
        _save_csv(ttest_result, "ttest_results")