"""
Multi-season feature builder (cross-season history).
Creates ONE combined CSV with the same columns as your per-season output,
for all seasons you choose. H2H and rolling stats look across previous years.

Author: ChatGPT
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict


# ====================================================
# CONFIG
# ====================================================
CONFIG = {
    # Folder containing your season CSVs. Files must be named like:
    #   2025_matches.csv, 2025_team_matches.csv, 2025_player_matches.csv
    "base_dir": "data/csv_datasets/all",
    # Seasons to include (edit this)
    "seasons": [2025],
    # Single combined output file
    "output_csv": "data/output/features_all_seasons.csv",
    # Rolling windows
    "win_size_form": 5,  # for form & team ratings
    "win_size_home_away": 5,  # for home/away xG & xGA
}


# ====================================================
# Helpers: robust column normalization
# ====================================================
def _rename_first_match(df, targets: dict):
    """Rename the first matching alias of each target to the standard name."""
    rename_map = {}
    for std, aliases in targets.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = std
                break
    return df.rename(columns=rename_map)


def normalize_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_first_match(
        df,
        {
            "fixture_id": ["fixture_id", "fixtureId", "match_id", "id"],
            "date_utc": ["date_utc", "utc_date", "kickoff_time", "date", "datetime"],
            "home_team_id": ["home_team_id", "homeTeamId", "home_id", "homeTeam_id"],
            "away_team_id": ["away_team_id", "awayTeamId", "away_id", "awayTeam_id"],
            "home_team": ["home_team", "homeTeam", "home_name", "home"],
            "away_team": ["away_team", "awayTeam", "away_name", "away"],
            "home_goals": ["home_goals", "home_goals_ft", "hg", "home_score"],
            "away_goals": ["away_goals", "away_goals_ft", "ag", "away_score"],
            "round": ["round", "stage", "phase"],
        },
    )
    if "date_utc" in df.columns:
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")
    for c in ["home_goals", "away_goals"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_team_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_first_match(
        df,
        {
            "fixture_id": ["fixture_id", "fixtureId", "match_id", "id"],
            "team_id": ["team_id", "teamId", "team", "team_code"],
            "is_home": ["is_home", "home", "at_home"],
            "expected_goals": ["expected_goals", "xG", "team_xg"],
            "goals_for": ["goals_for", "goals", "gf"],
            "goals_against": ["goals_against", "ga", "conceded"],
            "date_utc": ["date_utc", "utc_date", "kickoff_time", "date", "datetime"],
        },
    )
    # Normalize boolean for is_home
    if "is_home" in df.columns and df["is_home"].dtype != bool:
        df["is_home"] = (
            df["is_home"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        if df["is_home"].isna().any():
            df["is_home"] = (
                pd.to_numeric(df["is_home"], errors="coerce")
                .fillna(0)
                .astype(int)
                .astype(bool)
            )
    # Numerics
    for c in ["expected_goals", "goals_for", "goals_against"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure date_utc is datetime if present
    if "date_utc" in df.columns:
        df["date_utc"] = pd.to_datetime(df["date_utc"], errors="coerce")

    return df


def normalize_player_matches_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_first_match(
        df,
        {
            "fixture_id": ["fixture_id", "fixtureId", "match_id", "id"],
            "team_id": ["team_id", "teamId", "team", "team_code"],
            "player_id": ["player_id", "playerId", "player"],
            "rating": ["rating", "player_rating"],
            "starter": ["starter", "is_starter", "starting", "start"],
            "minutes": ["minutes", "mins", "time_played"],
            "team_name": [
                "team_name",
                "team",
                "club",
                "squad",
                "teamTitle",
                "teamName",
            ],
        },
    )
    df["rating"] = pd.to_numeric(df.get("rating"), errors="coerce")
    # starter: if missing, infer from minutes>0
    if "starter" not in df.columns:
        df["starter"] = (
            pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0) > 0
        ).astype(int)
    else:
        df["starter"] = (
            pd.to_numeric(df["starter"], errors="coerce").fillna(0).astype(int)
        )
    # Ensure minutes numeric (used later)
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    return df


# ====================================================
# Core builder over ALL seasons (chronological)
# ====================================================
def build_features_all(cfg):
    # 1) Load and normalize all seasons, then concatenate chronologically
    all_matches, all_team_matches, all_player_matches = [], [], []
    for season in cfg["seasons"]:
        print(f" Loading season {season}...")
        m_path = os.path.join(cfg["base_dir"], f"{season}_matches.csv")
        tm_path = os.path.join(cfg["base_dir"], f"{season}_team_matches.csv")
        pm_path = os.path.join(cfg["base_dir"], f"{season}_player_matches.csv")

        if not os.path.exists(m_path):
            raise FileNotFoundError(f"Missing file: {m_path}")
        if not os.path.exists(tm_path):
            raise FileNotFoundError(f"Missing file: {tm_path}")
        if not os.path.exists(pm_path):
            raise FileNotFoundError(f"Missing file: {pm_path}")

        m = normalize_matches_df(pd.read_csv(m_path))
        tm = normalize_team_matches_df(pd.read_csv(tm_path))
        pm = normalize_player_matches_df(pd.read_csv(pm_path))

        # Sanity: must have fixture_id + date for matches
        if "fixture_id" not in m.columns:
            raise KeyError(f"{season}_matches.csv missing 'fixture_id'")
        if "date_utc" not in m.columns:
            raise KeyError(
                f"{season}_matches.csv missing 'date_utc' (or alias); check CSV headers."
            )

        m["season"] = season
        tm["season"] = season
        pm["season"] = season
        all_matches.append(m)
        all_team_matches.append(tm)
        all_player_matches.append(pm)

    matches = pd.concat(all_matches, ignore_index=True).sort_values("date_utc")
    team_matches = pd.concat(all_team_matches, ignore_index=True)
    player_matches = pd.concat(all_player_matches, ignore_index=True)

    # ---------- Backfill team_matches.team_id from is_home + matches if needed ----------
    if ("team_id" not in team_matches.columns) or (
        team_matches["team_id"].isna().all()
    ):
        team_matches = team_matches.merge(
            matches[["fixture_id", "home_team_id", "away_team_id"]],
            on="fixture_id",
            how="left",
        )
        if "is_home" not in team_matches.columns:
            raise ValueError(
                "team_matches lacks both team_id and is_home; cannot infer team_id."
            )
        team_matches["team_id"] = np.where(
            team_matches["is_home"],
            team_matches["home_team_id"],
            team_matches["away_team_id"],
        )
        team_matches = team_matches.drop(
            columns=[
                c for c in ["home_team_id", "away_team_id"] if c in team_matches.columns
            ]
        )

    # ---------- Attach match dates for sorting team_matches chronologically (no column clash) ----------
    # If team_matches already has date_utc, just use it; otherwise pull from matches.
    if "date_utc" in team_matches.columns:
        team_matches["date_utc"] = pd.to_datetime(
            team_matches["date_utc"], errors="coerce"
        )
    else:
        team_matches = team_matches.merge(
            matches[["fixture_id", "date_utc"]], on="fixture_id", how="left"
        )

    # Sort by date and then drop the helper column (we don't need it further in team_matches)
    team_matches = team_matches.sort_values("date_utc").drop(columns=["date_utc"])

    # 2) Build features across full timeline (cross-season)

    # 2.1 Opponent xG WITHOUT self-merge (robust)
    tm = team_matches.copy()
    tm["expected_goals"] = pd.to_numeric(tm.get("expected_goals"), errors="coerce")
    fx_sum_xg = tm.groupby("fixture_id")["expected_goals"].transform("sum")
    tm["opp_expected_goals"] = fx_sum_xg - tm["expected_goals"]

    # Bring match dates (fresh) for chronological ops
    tm = (
        tm.merge(matches[["fixture_id", "date_utc"]], on="fixture_id", how="left")
        .sort_values(["team_id", "date_utc"])
        .reset_index(drop=True)
    )

    # 2.2 Points (3/1/0)
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

    # 2.3 Form (rolling last 5), shifted to avoid leakage
    w_form = cfg["win_size_form"]
    tm["form_points_last5"] = (
        tm.groupby("team_id")["points"]
        .rolling(w_form, min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # 2.4 Rolling xG/xGA (home/away) with transform, shifted
    w = cfg["win_size_home_away"]
    expg, oexpg, tid = tm["expected_goals"], tm["opp_expected_goals"], tm["team_id"]

    tm["home_xG_avg_team"] = (
        expg.where(tm["is_home"])
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )
    tm["away_xG_avg_team"] = (
        expg.where(~tm["is_home"])
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )
    tm["home_xGA_avg_team"] = (
        oexpg.where(tm["is_home"])
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )
    tm["away_xGA_avg_team"] = (
        oexpg.where(~tm["is_home"])
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )

    # 2.5 Win streaks (home/away), shifted
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

    # === Preserve team_id through groupby.apply across pandas versions
    tm["_team_id_backup"] = tm["team_id"]
    try:
        tm = tm.groupby("team_id", group_keys=False).apply(
            _streaks, include_groups=False
        )
    except TypeError:
        tm = tm.groupby("team_id", group_keys=False).apply(_streaks)
    if "team_id" not in tm.columns:
        tm = tm.reset_index(drop=False)
        if "team_id" not in tm.columns and "_team_id_backup" in tm.columns:
            tm["team_id"] = tm["_team_id_backup"]
    if "_team_id_backup" in tm.columns:
        tm = tm.drop(columns=["_team_id_backup"])

    # 2.6 TEAM RATING (robust to missing team_id in player_matches)
    pm = player_matches.copy()
    pm["rating"] = pd.to_numeric(pm.get("rating"), errors="coerce")
    pm["starter"] = pd.to_numeric(pm["starter"], errors="coerce").fillna(0).astype(int)
    if "minutes" in pm.columns:
        pm["minutes"] = pd.to_numeric(pm["minutes"], errors="coerce").fillna(0)

    def _has_team_id_ready(df: pd.DataFrame) -> bool:
        return ("team_id" in df.columns) and df["team_id"].notna().any()

    pm_has_team_id = _has_team_id_ready(pm)
    if not pm_has_team_id:
        tm_names = team_matches.merge(
            matches[
                ["fixture_id", "home_team_id", "home_team", "away_team_id", "away_team"]
            ],
            on="fixture_id",
            how="left",
        )
        tm_names["team_name"] = np.where(
            tm_names["team_id"] == tm_names["home_team_id"],
            tm_names["home_team"],
            np.where(
                tm_names["team_id"] == tm_names["away_team_id"],
                tm_names["away_team"],
                np.nan,
            ),
        )
        name_cols = [
            c
            for c in ["team_name", "team", "club", "squad", "teamTitle", "teamName"]
            if c in pm.columns
        ]
        if name_cols:
            nm = name_cols[0]
            if nm != "team_name":
                pm = pm.rename(columns={nm: "team_name"})
            pm = pm.merge(
                tm_names[["fixture_id", "team_id", "team_name"]],
                on=["fixture_id", "team_name"],
                how="left",
                validate="m:1",
            )
            pm_has_team_id = _has_team_id_ready(pm)

    eligible = pm[(pm["starter"] == 1) & (pm.get("team_id").notna())]
    can_rate = pm_has_team_id and (not eligible.empty)

    if can_rate:
        per_match_team_rating = (
            eligible.groupby(["fixture_id", "team_id"], as_index=False)["rating"]
            .mean()
            .rename(columns={"rating": "team_avg_player_rating"})
        )

        if "team_id" not in tm.columns:
            raise KeyError(
                "Internal: tm lost 'team_id' before ratings merge. "
                "Check groupby/apply and ensure restoration ran."
            )

        tm = tm.merge(per_match_team_rating, on=["fixture_id", "team_id"], how="left")
        tm["team_rating"] = (
            tm.groupby("team_id")["team_avg_player_rating"]
            .rolling(w_form, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
    else:
        tm["team_avg_player_rating"] = np.nan
        tm["team_rating"] = np.nan

    # 2.7 Rest days
    tm["rest_days"] = (
        tm.groupby("team_id")["date_utc"].diff().dt.total_seconds() / 86400
    )

    # 2.8 Assemble per fixture (home/away split)
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
            "season",
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

    # 2.9 Head-to-head (last 5 BEFORE current match) — across seasons
    ms = matches[
        [
            "fixture_id",
            "date_utc",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
        ]
    ].copy()

    def _h2h_counts(df, ht, at, date):
        past = (
            df[
                (df["date_utc"] < date)
                & (
                    ((df["home_team_id"] == ht) & (df["away_team_id"] == at))
                    | ((df["home_team_id"] == at) & (df["away_team_id"] == ht))
                )
            ]
            .sort_values("date_utc")
            .tail(5)
        )
        home_wins = (
            ((past["home_team_id"] == ht) & (past["home_goals"] > past["away_goals"]))
            | ((past["away_team_id"] == ht) & (past["away_goals"] > past["home_goals"]))
        ).sum()
        away_wins = (
            ((past["home_team_id"] == at) & (past["home_goals"] > past["away_goals"]))
            | ((past["away_team_id"] == at) & (past["away_goals"] > past["home_goals"]))
        ).sum()
        return int(home_wins), int(away_wins)

    h2h_home, h2h_away = [], []
    for _, r in feat.reset_index().iterrows():
        h, a = _h2h_counts(ms, r["home_team_id"], r["away_team_id"], r["date_utc"])
        h2h_home.append(h)
        h2h_away.append(a)
    feat["h2h_home_wins"] = h2h_home
    feat["h2h_away_wins"] = h2h_away

    # 2.10 Key players missing (top-5 by minutes + 10*avg_rating, accumulated BEFORE match) — cross-season
    pm_small = player_matches[
        ["fixture_id", "team_id", "player_id", "minutes", "rating"]
    ].copy()
    pm_small["rating"] = pd.to_numeric(pm_small["rating"], errors="coerce")
    pm_small["minutes"] = pd.to_numeric(
        pm_small.get("minutes"), errors="coerce"
    ).fillna(0)
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
        for role, tid in [("home", row["home_team_id"]), ("away", row["away_team_id"])]:
            ps = cum_stats[tid]
            scores = [
                (
                    pid,
                    v["minutes"]
                    + 10 * ((v["rating_sum"] / v["apps"]) if v["apps"] else 0),
                )
                for pid, v in ps.items()
            ]
            top5 = set(pid for pid, _ in sorted(scores, key=lambda x: -x[1])[:5])
            lineup = lineup_ids.get((fx, tid), set())
            missing = len([pid for pid in top5 if pid not in lineup]) if top5 else 0
            if role == "home":
                key_home.append((fx, missing))
            else:
                key_away.append((fx, missing))
        # update cumulative stats AFTER match
        for tid, grp in pm_small[pm_small["fixture_id"] == fx].groupby("team_id"):
            for _, rr in grp.iterrows():
                s = cum_stats[tid][rr["player_id"]]
                s["minutes"] += rr["minutes"] if pd.notna(rr["minutes"]) else 0
                if pd.notna(rr["rating"]):
                    s["rating_sum"] += rr["rating"]
                s["apps"] += 1

    feat = feat.join(
        pd.DataFrame(
            key_home, columns=["fixture_id", "key_players_missing_home"]
        ).set_index("fixture_id")
    )
    feat = feat.join(
        pd.DataFrame(
            key_away, columns=["fixture_id", "key_players_missing_away"]
        ).set_index("fixture_id")
    )

    # 2.11 Match importance
    def _importance(r):
        if isinstance(r, str) and any(
            k in r.lower() for k in ["final", "semi", "quarter"]
        ):
            return 1.2
        return 1.0

    feat["match_importance"] = feat["round"].apply(_importance)
    feat = feat.drop(columns=["round"]).reset_index()

    # 3) Save single combined CSV
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    feat = feat.sort_values(["date_utc", "fixture_id"])
    feat.to_csv(cfg["output_csv"], index=False)

    print(f" Combined features saved to: {cfg['output_csv']}")
    print(f"Rows: {len(feat):,} | Seasons: {sorted(matches['season'].unique())}")
    return feat


# ====================================================
# Run
# ====================================================
if __name__ == "__main__":
    build_features_all(CONFIG)
