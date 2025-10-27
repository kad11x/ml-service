# Genetic Algorithm for Feature Selection using Log Loss (H/D/A prediction)
# - Reads features_all_seasons.csv
# - Derives H/D/A from home_goals vs away_goals
# - Uses LogisticRegression and mean CV log loss as GA fitness (lower is better)
# - Avoids leakage by excluding home_goals/away_goals from features
# - Saves GA history and best subset to Testing/data (OUT_DIR="data")

import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------

CSV_PATH = "../data/output/features_all_seasons.csv"  # change if needed
OUT_DIR = "data"  # saves to Testing/data since this script is in Testing/

# GA hyperparameters
population_size = 100
generations = 30
crossover_rate = 0.8
mutation_rate = 0.1
tournament_size = 3
random_seed = 123

# -----------------------
# Load & prepare data
# -----------------------
df = pd.read_csv(CSV_PATH).copy()
print("Loaded:", CSV_PATH, "| shape:", df.shape)


def derive_result(row):
    if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
        return np.nan
    if row["home_goals"] > row["away_goals"]:
        return "H"
    elif row["home_goals"] < row["away_goals"]:
        return "A"
    else:
        return "D"


if "home_goals" not in df.columns or "away_goals" not in df.columns:
    raise ValueError(
        "CSV must contain 'home_goals' and 'away_goals' to derive H/D/A target."
    )

df["result_HDA"] = df.apply(derive_result, axis=1)
print("Result distribution:\n", df["result_HDA"].value_counts(dropna=False))

# Candidate numeric features (exclude identifiers, team names, target, and leakage columns)
exclude_cols = {
    "fixture_id",
    "date_utc",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "result_HDA",
    "home_goals",
    "away_goals",
}
numeric_candidates = [
    c
    for c in df.columns
    if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols
]


def non_constant_columns(frame: pd.DataFrame, cols: List[str]) -> List[str]:
    keep = []
    for c in cols:
        s = frame[c].dropna().values
        if len(s) == 0:  # all NaN
            continue
        if np.nanstd(s) == 0:  # zero variance
            continue
        keep.append(c)
    return keep


feature_candidates = non_constant_columns(df, numeric_candidates)
print(f"Candidate features: {len(feature_candidates)}")
if len(feature_candidates) == 0:
    raise ValueError("No usable numeric features found after filtering.")

# Keep only rows with target
mask = df["result_HDA"].notna()
X = df.loc[mask, feature_candidates].copy()
y = df.loc[mask, "result_HDA"].astype(str)

# Fixed class order for consistent log_loss across folds
Y_CLASSES = np.array(sorted(y.unique()))


# -----------------------
# Fitness: Stratified K-Fold CV log loss
# -----------------------
def evaluate_subset_cv(
    cols: List[str], X_df: pd.DataFrame, y_series: pd.Series, n_splits: int = 5
) -> float:
    """Mean stratified K-fold log loss for a given subset of columns (lower is better)."""
    if len(cols) == 0:
        return 5.0  # penalty for empty subset

    X_vals = X_df[cols].values
    y_vals = y_series.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    losses = []

    for tr_idx, va_idx in skf.split(X_vals, y_vals):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y_vals[tr_idx], y_vals[va_idx]

        try:
            pipe = Pipeline(
                steps=[
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(max_iter=1000),
                    ),  # add class_weight="balanced" if needed
                ]
            )

            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_va)

            # Reorder columns of proba to match Y_CLASSES for log_loss
            cls_order = pipe.named_steps["clf"].classes_
            aligned = np.zeros((proba.shape[0], len(Y_CLASSES)), dtype=float)
            for j, cls in enumerate(cls_order):
                aligned[:, np.where(Y_CLASSES == cls)[0][0]] = proba[:, j]

            fold_loss = log_loss(y_va, aligned, labels=Y_CLASSES)
        except Exception:
            fold_loss = 5.0  # penalize numerical/convergence issues

        losses.append(fold_loss)

    return float(np.mean(losses))


# -----------------------
# Genetic Algorithm
# -----------------------
rng = np.random.RandomState(random_seed)
n_features = len(feature_candidates)


def random_bits() -> np.ndarray:
    b = rng.randint(0, 2, size=n_features)
    if b.sum() == 0:
        b[rng.randint(0, n_features)] = 1
    return b


def bits_to_cols(bits: np.ndarray) -> List[str]:
    return [feature_candidates[i] for i, v in enumerate(bits) if v == 1]


def bits_key(bits: np.ndarray) -> str:
    return "".join(map(str, bits.tolist()))


def tournament_select(pop: List[np.ndarray], fits: List[float]) -> np.ndarray:
    idxs = rng.choice(len(pop), size=tournament_size, replace=False)
    j = min(idxs, key=lambda i: fits[i])  # lower is better
    return pop[j].copy()


def crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rng.rand() > crossover_rate or n_features < 2:
        return a.copy(), b.copy()
    p = rng.randint(1, n_features)
    c1 = np.concatenate([a[:p], b[p:]])
    c2 = np.concatenate([b[:p], a[p:]])
    # enforce at least one 1
    if c1.sum() == 0:
        c1[rng.randint(0, n_features)] = 1
    if c2.sum() == 0:
        c2[rng.randint(0, n_features)] = 1
    return c1, c2


def mutate(bits: np.ndarray) -> np.ndarray:
    for i in range(n_features):
        if rng.rand() < mutation_rate:
            bits[i] = 1 - bits[i]
    if bits.sum() == 0:
        bits[rng.randint(0, n_features)] = 1
    return bits


# Initialize population
population = [random_bits() for _ in range(population_size)]
fitness_cache: Dict[str, float] = {}
history_rows = []

best_bits = None
best_ll = float("inf")

for g in range(generations):
    fitnesses = []
    for b in population:
        k = bits_key(b)
        if k in fitness_cache:
            ll = fitness_cache[k]
        else:
            cols = bits_to_cols(b)
            ll = evaluate_subset_cv(cols, X, y, n_splits=5)
            fitness_cache[k] = ll
        fitnesses.append(ll)

    # Track generation stats
    g_best_idx = int(np.argmin(fitnesses))
    g_best_bits = population[g_best_idx].copy()
    g_best_ll = float(fitnesses[g_best_idx])

    if g_best_ll < best_ll:
        best_ll = g_best_ll
        best_bits = g_best_bits.copy()

    history_rows.append(
        {
            "generation": g,
            "best_log_loss": g_best_ll,
            "mean_log_loss": float(np.mean(fitnesses)),
            "median_log_loss": float(np.median(fitnesses)),
        }
    )
    print(
        f"Gen {g:02d} | best {g_best_ll:.4f} | mean {np.mean(fitnesses):.4f} | median {np.median(fitnesses):.4f}"
    )

    # Next generation: elitism + tournament + crossover + mutation
    next_pop = [g_best_bits.copy()]
    while len(next_pop) < population_size:
        p1 = tournament_select(population, fitnesses)
        p2 = tournament_select(population, fitnesses)
        c1, c2 = crossover(p1, p2)
        next_pop.append(mutate(c1))
        if len(next_pop) < population_size:
            next_pop.append(mutate(c2))
    population = next_pop

# -----------------------
# Results & Save
# -----------------------
best_cols = bits_to_cols(best_bits)
print("\n=== GA Summary ===")
print("Best CV log loss:", best_ll)
print("Best number of features:", len(best_cols))
print("Best subset:")
for c in best_cols:
    print(" -", c)

os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(history_rows).to_csv(os.path.join(OUT_DIR, "ga_history.csv"), index=False)
with open(os.path.join(OUT_DIR, "best_subset.txt"), "w") as f:
    f.write("\n".join(best_cols))

print("\nSaved:")
print(" -", os.path.join(OUT_DIR, "ga_history.csv"))
print(" -", os.path.join(OUT_DIR, "best_subset.txt"))


hist = pd.read_csv("data/ga_history.csv")
plt.plot(hist["generation"], hist["best_log_loss"], label="Best")
plt.plot(hist["generation"], hist["mean_log_loss"], label="Mean")
plt.legend()
plt.show()
