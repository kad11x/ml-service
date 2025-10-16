"""
Feature Engineering Script for Football Match Prediction

Generates a CSV file with advanced match-level features such as:
xG averages, team ratings, form, streaks, head-to-head stats,
rest days, key player absences, and match importance.


"""

import pandas as pd
import numpy as np
from collections import defaultdict

# ============================================================
# ⚙️ CONFIGURATION SECTION
# ============================================================

CONFIG = {
    # Input CSV file paths
    "matches_csv": "data/csv_datasets/epl/2025_matches.csv",
    "team_matches_csv": "data/csv_datasets/epl/2025_team_matches.csv",
    "player_matches_csv": "data/csv_datasets/epl/2025_player_matches.csv",
    "output_csv": "data/output/features_2025_all.csv",
    # Rolling window sizes
    "win_size_form": 5,  # for form points and team ratings
    "win_size_home_away": 5,  # for home/away xG and xGA averages
}
# ============================================================


def build_features(cfg: dict):
    # ---------- LOAD ----------
    matches = pd.read_csv(cfg["matches_csv"], parse_dates=["date_utc"])
    team_matches = pd.read_csv(cfg["team_matches_csv"])
    player_matches = pd.read_csv(cfg["player_matches_csv"])

    # Ensure goal cols numeric
    for col in ["home_goals", "away_goals"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    # Ensure is_home is boolean (robust to 0/1/"True"/"False")
    if team_matches["is_home"].dtype != bool:
        team_matches["is_home"] = (
            team_matches["is_home"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
            .fillna(team_matches["is_home"])
        )
        # If still not boolean, try numeric->bool
        if team_matches["is_home"].dtype != bool:
            team_matches["is_home"] = (
                pd.to_numeric(team_matches["is_home"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(bool)
            )

    # ---------- ADD OPPONENT xG ----------
    tm = team_matches.copy()
    tm = tm.merge(
        tm[["fixture_id", "team_id", "expected_goals"]].rename(
            columns={"team_id": "opp_team_id", "expected_goals": "opp_expected_goals"}
        ),
        on="fixture_id",
        how="left",
    )
    tm = tm[tm["team_id"] != tm["opp_team_id"]].copy()

    # Join date & sort for rolling calcs
    tm = tm.merge(matches[["fixture_id", "date_utc"]], on="fixture_id", how="left")
    tm = tm.sort_values(["team_id", "date_utc"]).reset_index(drop=True)

    # ---------- RESULT → POINTS ----------
    def _points(row):
        gf, ga = row["goals_for"], row["goals_against"]
        if pd.isna(gf) or pd.isna(ga):
            return np.nan
        if gf > ga:
            return 3
        if gf == ga:
            return 1
        return 0

    tm["points"] = tm.apply(_points, axis=1)

    # ---------- FORM (last 5, shifted) ----------
    tm["form_points_last5"] = (
        tm.groupby("team_id")["points"]
        .rolling(cfg["win_size_form"], min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # ---------- ROLLING HELPERS USING TRANSFORM (index-safe) ----------
    w = cfg["win_size_home_away"]

    # Expected goals for (home/away)
    # We mask to NaN outside the venue we care about, then do groupby(...).transform(...)
    expg = tm["expected_goals"]
    opp_expg = tm["opp_expected_goals"]
    team_key = tm["team_id"]

    tm["home_xG_avg_team"] = (
        expg.where(tm["is_home"])
        .groupby(team_key)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )

    tm["away_xG_avg_team"] = (
        expg.where(~tm["is_home"])
        .groupby(team_key)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )

    tm["home_xGA_avg_team"] = (
        opp_expg.where(tm["is_home"])
        .groupby(team_key)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )

    tm["away_xGA_avg_team"] = (
        opp_expg.where(~tm["is_home"])
        .groupby(team_key)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )

    # ---------- WIN STREAKS (shifted) ----------
    def _streaks(g):
        cur_h = cur_a = 0
        h, a = [], []
        for _, r in g.iterrows():
            if r["is_home"]:
                cur_h = cur_h + 1 if r["goals_for"] > r["goals_against"] else 0
                h.append(cur_h)
                a.append(np.nan)
            else:
                cur_a = cur_a + 1 if r["goals_for"] > r["goals_against"] else 0
                a.append(cur_a)
                h.append(np.nan)
        g["home_win_streak_team"] = pd.Series(h, index=g.index).shift(1)
        g["away_win_streak_team"] = pd.Series(a, index=g.index).shift(1)
        return g

    tm = tm.groupby("team_id", group_keys=False).apply(_streaks)

    # ---------- TEAM RATING (avg starters’ rating → rolling last 5, shifted) ----------
    pm = player_matches.copy()
    pm["rating"] = pd.to_numeric(pm["rating"], errors="coerce")
    pm["starter"] = pd.to_numeric(pm["starter"], errors="coerce").fillna(0).astype(int)
    team_rating = (
        pm[pm["starter"] == 1]
        .groupby(["fixture_id", "team_id"], as_index=False)["rating"]
        .mean()
        .rename(columns={"rating": "team_avg_player_rating"})
    )
    tm = tm.merge(team_rating, on=["fixture_id", "team_id"], how="left")
    tm["team_rating"] = (
        tm.groupby("team_id")["team_avg_player_rating"]
        .rolling(cfg["win_size_form"], min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # ---------- REST DAYS ----------
    tm["rest_days"] = (
        tm.groupby("team_id")["date_utc"].diff().dt.total_seconds() / 86400
    )

    # ---------- MERGE HOME/AWAY VIEW ----------
    base = matches[
        [
            "fixture_id",
            "date_utc",
            "home_team_id",
            "home_team",
            "away_team_id",
            "away_team",
            "home_goals",
            "away_goals",
            "round",
        ]
    ].set_index("fixture_id")

    home_tm = tm[tm["is_home"]].set_index("fixture_id")
    away_tm = tm[~tm["is_home"]].set_index("fixture_id")

    feat = base.copy()
    feat["home_xG_avg"] = home_tm["home_xG_avg_team"]
    feat["away_xG_avg"] = away_tm["away_xG_avg_team"]
    feat["home_xGA_avg"] = home_tm["home_xGA_avg_team"]
    feat["away_xGA_avg"] = away_tm["away_xGA_avg_team"]
    feat["home_team_rating"] = home_tm["team_rating"]
    feat["away_team_rating"] = away_tm["team_rating"]
    feat["home_form_points"] = home_tm["form_points_last5"]
    feat["away_form_points"] = away_tm["form_points_last5"]
    feat["home_win_streak"] = home_tm["home_win_streak_team"]
    feat["away_win_streak"] = away_tm["away_win_streak_team"]
    feat["rest_days_home"] = home_tm["rest_days"]
    feat["rest_days_away"] = away_tm["rest_days"]

    # ---------- HEAD-TO-HEAD (last 5 before fixture) ----------
    matches_small = matches[
        [
            "fixture_id",
            "date_utc",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
        ]
    ].copy()

    def _h2h_counts(df, home_team_id, away_team_id, date):
        past = (
            df[
                (df["date_utc"] < date)
                & (
                    (
                        (df["home_team_id"] == home_team_id)
                        & (df["away_team_id"] == away_team_id)
                    )
                    | (
                        (df["home_team_id"] == away_team_id)
                        & (df["away_team_id"] == home_team_id)
                    )
                )
            ]
            .sort_values("date_utc")
            .tail(5)
        )

        home_wins = (
            (
                (past["home_team_id"] == home_team_id)
                & (past["home_goals"] > past["away_goals"])
            )
            | (
                (past["away_team_id"] == home_team_id)
                & (past["away_goals"] > past["home_goals"])
            )
        ).sum()

        away_wins = (
            (
                (past["home_team_id"] == away_team_id)
                & (past["home_goals"] > past["away_goals"])
            )
            | (
                (past["away_team_id"] == away_team_id)
                & (past["away_goals"] > past["home_goals"])
            )
        ).sum()

        return int(home_wins), int(away_wins)

    h2h_home, h2h_away = [], []
    for _, r in feat.reset_index().iterrows():
        h, a = _h2h_counts(
            matches_small, r["home_team_id"], r["away_team_id"], r["date_utc"]
        )
        h2h_home.append(h)
        h2h_away.append(a)
    feat["h2h_home_wins"] = h2h_home
    feat["h2h_away_wins"] = h2h_away

    # ---------- KEY PLAYERS MISSING ----------
    pm_small = player_matches[
        ["fixture_id", "team_id", "player_id", "minutes", "rating"]
    ].copy()
    pm_small["rating"] = pd.to_numeric(pm_small["rating"], errors="coerce")
    pm_small = pm_small.merge(
        matches[["fixture_id", "date_utc"]], on="fixture_id", how="left"
    ).sort_values("date_utc")

    lineup_ids = (
        pm_small.groupby(["fixture_id", "team_id"])["player_id"].apply(set).to_dict()
    )
    cum_stats = defaultdict(
        lambda: defaultdict(lambda: {"minutes": 0.0, "rating_sum": 0.0, "apps": 0})
    )

    key_home, key_away = [], []
    for _, row in (
        matches[["fixture_id", "date_utc", "home_team_id", "away_team_id"]]
        .sort_values("date_utc")
        .iterrows()
    ):
        fx = row["fixture_id"]

        for role, team_id in [
            ("home", row["home_team_id"]),
            ("away", row["away_team_id"]),
        ]:
            player_stats = cum_stats[team_id]
            scores = [
                (
                    pid,
                    v["minutes"]
                    + 10 * ((v["rating_sum"] / v["apps"]) if v["apps"] else 0),
                )
                for pid, v in player_stats.items()
            ]
            top5 = set(pid for pid, _ in sorted(scores, key=lambda x: -x[1])[:5])
            lineup = lineup_ids.get((fx, team_id), set())
            missing = len([pid for pid in top5 if pid not in lineup]) if top5 else 0
            if role == "home":
                key_home.append((fx, missing))
            else:
                key_away.append((fx, missing))

        # update cumulative stats after match
        for team_id, grp in pm_small[pm_small["fixture_id"] == fx].groupby("team_id"):
            for _, rr in grp.iterrows():
                s = cum_stats[team_id][rr["player_id"]]
                s["minutes"] += rr["minutes"] if pd.notna(rr["minutes"]) else 0
                if pd.notna(rr["rating"]):
                    s["rating_sum"] += rr["rating"]
                s["apps"] += 1

    key_home_df = pd.DataFrame(
        key_home, columns=["fixture_id", "key_players_missing_home"]
    ).set_index("fixture_id")
    key_away_df = pd.DataFrame(
        key_away, columns=["fixture_id", "key_players_missing_away"]
    ).set_index("fixture_id")
    feat = feat.join(key_home_df).join(key_away_df)

    # ---------- MATCH IMPORTANCE ----------
    def _importance(r):
        if isinstance(r, str) and any(
            k in r.lower() for k in ["final", "semi", "quarter"]
        ):
            return 1.2
        return 1.0

    feat["match_importance"] = feat["round"].apply(_importance)
    feat = feat.drop(columns=["round"])

    # ---------- SAVE ----------
    feat.to_csv(cfg["output_csv"], index=True)
    print(f" Features saved to: {cfg['output_csv']}")
    return feat


if __name__ == "__main__":
    build_features(CONFIG)
