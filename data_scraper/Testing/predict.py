"""
Predict H/D/A using each team's latest available data (NOT last head-to-head).

Flow:
1. Load features_all_seasons.csv
2. Train on 6 GA features
3. In a loop: ask for home + away team
4. For the home team: find its latest match where it was HOME
   - if none, take latest match regardless of venue and adapt
5. For the away team: find its latest match where it was AWAY
   - if none, take latest match regardless of venue and adapt
6. Build ONE synthetic row with:
   home_team_rating  <- from home team's latest row
   home_form_points  <- from home team's latest row
   rest_days_away    <- from AWAY team's latest row
   away_team_rating  <- from away team's latest row
   h2h_*             <- can be taken from latest head-to-head, or 0 if no h2h
7. Predict, show percentages
"""

import pandas as pd
import numpy as np
import re

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

CSV_PATH = "../data/output/features_all_seasons.csv"  # change if needed

GA_FEATURES = [
    "home_team_rating",
    "away_team_rating",
    "home_form_points",
    "rest_days_away",
    "h2h_home_wins",
    "h2h_away_wins",
]


# -------------------------
# name normalization
# -------------------------
def norm_name(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b(fc|afc)\b", "", s)
    s = s.replace(".", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# -------------------------
# load + train
# -------------------------
print("⚙️  Loading data...")
df = pd.read_csv(CSV_PATH)


# target
def derive_result(row):
    if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
        return None
    if row["home_goals"] > row["away_goals"]:
        return "H"
    elif row["home_goals"] < row["away_goals"]:
        return "A"
    else:
        return "D"


df["result_HDA"] = df.apply(derive_result, axis=1)
mask = df["result_HDA"].notna()

# normalized columns for lookup
df["home_team_norm"] = df["home_team"].astype(str).apply(norm_name)
df["away_team_norm"] = df["away_team"].astype(str).apply(norm_name)

# make sure dates are datetime
if "date_utc" in df.columns:
    df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")

X = df.loc[mask, GA_FEATURES]
y = df.loc[mask, "result_HDA"].astype(str)

model = Pipeline(
    [
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
model.fit(X, y)
CLASSES = model.named_steps["clf"].classes_

print("✅ Model trained.\n")


# -------------------------
# helpers to get latest rows
# -------------------------
def get_latest_home_row(df: pd.DataFrame, team: str):
    team_n = norm_name(team)
    cand = df[df["home_team_norm"] == team_n].copy()
    if cand.empty:
        return None
    cand = cand.sort_values("date_utc")
    return cand.iloc[-1]


def get_latest_away_row(df: pd.DataFrame, team: str):
    team_n = norm_name(team)
    cand = df[df["away_team_norm"] == team_n].copy()
    if cand.empty:
        return None
    cand = cand.sort_values("date_utc")
    return cand.iloc[-1]


def get_latest_any_row(df: pd.DataFrame, team: str):
    team_n = norm_name(team)
    cand = df[
        (df["home_team_norm"] == team_n) | (df["away_team_norm"] == team_n)
    ].copy()
    if cand.empty:
        return None
    cand = cand.sort_values("date_utc")
    return cand.iloc[-1]


def get_latest_h2h(df: pd.DataFrame, home_team: str, away_team: str):
    """Optional: grab h2h from latest meeting to fill h2h_home_wins / h2h_away_wins.
    If none, return (0, 0)."""
    h = norm_name(home_team)
    a = norm_name(away_team)
    cand = df[
        ((df["home_team_norm"] == h) & (df["away_team_norm"] == a))
        | ((df["home_team_norm"] == a) & (df["away_team_norm"] == h))
    ].copy()
    if cand.empty:
        return 0.0, 0.0
    cand = cand.sort_values("date_utc")
    row = cand.iloc[-1]
    # if orientation matches
    if norm_name(row["home_team"]) == h and norm_name(row["away_team"]) == a:
        return float(row.get("h2h_home_wins", 0.0)), float(
            row.get("h2h_away_wins", 0.0)
        )
    else:
        # swap
        return float(row.get("h2h_away_wins", 0.0)), float(
            row.get("h2h_home_wins", 0.0)
        )


# -------------------------
# build synthetic match from LATEST per team
# -------------------------
def build_synthetic_row(
    df: pd.DataFrame, home_team: str, away_team: str
) -> pd.DataFrame:
    # 1) latest HOME data for home team
    home_row = get_latest_home_row(df, home_team)
    if home_row is None:
        # fallback: any match
        home_row = get_latest_any_row(df, home_team)
    if home_row is None:
        raise ValueError(f"No data in CSV for home team '{home_team}'")

    # 2) latest AWAY data for away team
    away_row = get_latest_away_row(df, away_team)
    if away_row is None:
        away_row = get_latest_any_row(df, away_team)
    if away_row is None:
        raise ValueError(f"No data in CSV for away team '{away_team}'")

    # 3) h2h (optional)
    h2h_home, h2h_away = get_latest_h2h(df, home_team, away_team)

    data = {
        "home_team_rating": (
            home_row.get("home_team_rating", np.nan)
            if norm_name(home_row["home_team"]) == norm_name(home_team)
            else home_row.get("away_team_rating", np.nan)
        ),
        "home_form_points": (
            home_row.get("home_form_points", np.nan)
            if "home_form_points" in home_row
            else np.nan
        ),
        "away_team_rating": (
            away_row.get("away_team_rating", np.nan)
            if norm_name(away_row["away_team"]) == norm_name(away_team)
            else away_row.get("home_team_rating", np.nan)
        ),
        "rest_days_away": (
            away_row.get("rest_days_away", np.nan)
            if "rest_days_away" in away_row
            else np.nan
        ),
        "h2h_home_wins": h2h_home,
        "h2h_away_wins": h2h_away,
    }

    return pd.DataFrame([data], columns=GA_FEATURES)


# -------------------------
# prediction
# -------------------------
def predict_today(df: pd.DataFrame, home_team: str, away_team: str):
    X_new = build_synthetic_row(df, home_team, away_team)
    proba = model.predict_proba(X_new)[0]
    pred = CLASSES[proba.argmax()]
    probs_percent = [p * 100 for p in proba]

    print(
        f"\n🔹 Prediction for {home_team} vs {away_team} (using each team's latest match)"
    )
    for cls, p in zip(CLASSES, probs_percent):
        print(f"  {cls}: {p:.1f}%")
    print(f"👉 Predicted result: {pred}\n")


# -------------------------
# interactive loop
# -------------------------
while True:
    home = input("🏠 Home team (or 'exit' to quit): ").strip()
    if home.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    away = input("🚗 Away team: ").strip()
    if away.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
        break

    try:
        predict_today(df, home, away)
    except ValueError as e:
        print("⚠️", e)
