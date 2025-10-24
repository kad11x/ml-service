"""
Multi-season feature builder (cross-season history).
Creates ONE combined CSV with the same columns as your per-season output,
for all seasons you choose. H2H and rolling stats look across previous years.

Author: ChatGPT
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Iterable, List


# ====================================================
# Helpers: robust column normalization
# ====================================================
def _rename_first_match(df, targets: dict):
    """Rename the first matching alias of each target to the standard name."""
    rename_map = {}
    lower = {c.lower(): c for c in df.columns}
    for std, aliases in targets.items():
        for a in aliases:
            if a.lower() in lower:
                rename_map[lower[a.lower()]] = std
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
            "is_home": ["is_home", "home", "at_home", "venue", "side"],
            # expected_goals may be absent; we calculate it if missing.
            "expected_goals": ["expected_goals", "xG", "xg", "team_xg"],
            "goals_for": ["goals_for", "goals", "gf"],
            "goals_against": ["goals_against", "ga", "conceded"],
            "date_utc": ["date_utc", "utc_date", "kickoff_time", "date", "datetime"],
        },
    )
    # Normalize boolean for is_home
    if "is_home" in df.columns and df["is_home"].dtype != bool:
        s = df["is_home"].astype(str).str.strip().str.lower()
        df["is_home"] = s.map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "home": True,
                "away": False,
                "h": True,
                "a": False,
            }
        )
        if df["is_home"].isna().any():
            df["is_home"] = (
                pd.to_numeric(s, errors="coerce").fillna(0).astype(int).astype(bool)
            )

    for c in ["expected_goals", "goals_for", "goals_against"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
    if "starter" not in df.columns:
        df["starter"] = (
            pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0) > 0
        ).astype(int)
    else:
        df["starter"] = (
            pd.to_numeric(df["starter"], errors="coerce").fillna(0).astype(int)
        )
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    return df


# ====================================================
# xG calculation from goals (no external xG needed)
# ====================================================
def _ensure_is_home(team_matches: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Infer is_home if missing by comparing team_id to matches.home_team_id."""
    if "is_home" in team_matches.columns:
        return team_matches
    if "team_id" not in team_matches.columns:
        return team_matches
    mm = matches[["fixture_id", "home_team_id"]]
    tm2 = team_matches.merge(mm, on="fixture_id", how="left")
    tm2["is_home"] = tm2["team_id"].eq(tm2["home_team_id"])
    return tm2.drop(columns=["home_team_id"], errors="ignore")


def _coalesce_date_utc(df: pd.DataFrame) -> pd.DataFrame:
    """After merges, collapse date_utc_x/date_utc_y into date_utc."""
    if "date_utc" in df.columns:
        return df
    left = pd.to_datetime(df.get("date_utc_x"), errors="coerce")
    right = pd.to_datetime(df.get("date_utc_y"), errors="coerce")
    if ("date_utc_x" not in df.columns) and ("date_utc_y" not in df.columns):
        return df
    df["date_utc"] = left
    if "date_utc_y" in df.columns:
        df["date_utc"] = df["date_utc"].where(df["date_utc"].notna(), right)
    return df.drop(columns=[c for c in ["date_utc_x", "date_utc_y"] if c in df.columns])


def add_calculated_expected_goals(
    team_matches: pd.DataFrame,
    matches: pd.DataFrame,
    win_size_home_away: int,
    min_matches_for_factors: int = 2,
) -> pd.DataFrame:
    """
    Compute per-team pre-match expected_goals using rolling GF/GA and
    season baselines (μ_home, μ_away), fully leakage-safe.
    Adds/overwrites column 'expected_goals' in team_matches.
    """
    w = int(win_size_home_away)
    min_mp = int(min_matches_for_factors)

    # Attach season & opponent ids and date to team rows (this merge can create date_utc_x/y)
    mm = matches[
        ["fixture_id", "date_utc", "season", "home_team_id", "away_team_id"]
    ].copy()
    tm = team_matches.merge(mm, on="fixture_id", how="left")

    # Coalesce date_utc_x / date_utc_y if created
    tm = _coalesce_date_utc(tm)

    # Ensure is_home present; infer if missing
    tm = _ensure_is_home(tm, matches)

    # Sort for rolling calcs
    tm = tm.sort_values(["team_id", "date_utc"]).reset_index(drop=True)

    # League baselines per season (constants per season)
    season_home_avg = (
        matches.groupby("season", dropna=False)["home_goals"].mean().rename("mu_home")
    )
    season_away_avg = (
        matches.groupby("season", dropna=False)["away_goals"].mean().rename("mu_away")
    )
    tm = tm.merge(season_home_avg, on="season", how="left")
    tm = tm.merge(season_away_avg, on="season", how="left")

    # Rolling GF/GA split by venue (shifted to avoid leakage)
    is_home = tm["is_home"].astype(bool)
    tid = tm["team_id"]

    tm["roll_home_gf"] = (
        tm["goals_for"]
        .where(is_home)
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )
    tm["roll_away_gf"] = (
        tm["goals_for"]
        .where(~is_home)
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )
    tm["roll_home_ga"] = (
        tm["goals_against"]
        .where(is_home)
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )
    tm["roll_away_ga"] = (
        tm["goals_against"]
        .where(~is_home)
        .groupby(tid)
        .transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    )

    # Match counts (to gate factors)
    tm["cnt_home"] = is_home.groupby(tid).transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    tm["cnt_away"] = (
        (~is_home).groupby(tid).transform(lambda s: s.cumsum().shift(1).fillna(0))
    )

    # Attack & defense factors (with clamps to avoid extremes)
    eps = 1e-9
    tm["attack_home_factor"] = np.where(
        (tm["cnt_home"] >= min_mp) & tm["mu_home"].notna(),
        (tm["roll_home_gf"] / (tm["mu_home"] + eps)).clip(0.1, 5.0),
        1.0,
    )
    tm["attack_away_factor"] = np.where(
        (tm["cnt_away"] >= min_mp) & tm["mu_away"].notna(),
        (tm["roll_away_gf"] / (tm["mu_away"] + eps)).clip(0.1, 5.0),
        1.0,
    )
    tm["defense_home_factor"] = np.where(
        (tm["cnt_home"] >= min_mp) & tm["mu_away"].notna(),
        (tm["roll_home_ga"] / (tm["mu_away"] + eps)).clip(0.1, 5.0),
        1.0,
    )
    tm["defense_away_factor"] = np.where(
        (tm["cnt_away"] >= min_mp) & tm["mu_home"].notna(),
        (tm["roll_away_ga"] / (tm["mu_home"] + eps)).clip(0.1, 5.0),
        1.0,
    )

    # Get opponent defensive factor per fixture
    opp = tm[
        [
            "fixture_id",
            "team_id",
            "is_home",
            "defense_home_factor",
            "defense_away_factor",
        ]
    ].copy()
    opp = opp.rename(
        columns={
            "team_id": "opp_team_id",
            "is_home": "opp_is_home",
            "defense_home_factor": "opp_defense_home_factor",
            "defense_away_factor": "opp_defense_away_factor",
        }
    )
    tm = tm.merge(opp, on="fixture_id", how="left")
    tm = tm[tm["team_id"] != tm["opp_team_id"]].copy()

    # Expected goals per team row (pre-match)
    tm["expected_goals"] = np.where(
        tm["is_home"],
        tm["mu_home"] * tm["attack_home_factor"] * tm["opp_defense_away_factor"],
        tm["mu_away"] * tm["attack_away_factor"] * tm["opp_defense_home_factor"],
    )

    # Fallback for earliest matches: season baseline by venue
    tm["expected_goals"] = np.where(
        tm["expected_goals"].notna(),
        tm["expected_goals"],
        np.where(tm["is_home"], tm["mu_home"], tm["mu_away"]),
    )

    # Return to team_matches shape
    tm_expected = tm[["fixture_id", "team_id", "expected_goals"]]
    out = team_matches.merge(
        tm_expected, on=["fixture_id", "team_id"], how="left", validate="m:1"
    )
    return out


# ====================================================
# Season argument parsing
# ====================================================
def _detect_seasons(base_dir: str) -> List[int]:
    """Detect available seasons by scanning for '<year>_matches.csv' (and companion files)."""
    if not os.path.isdir(base_dir):
        return []
    years = []
    for fn in os.listdir(base_dir):
        m = re.match(r"^(\d{4})_matches\.csv$", fn)
        if not m:
            continue
        y = int(m.group(1))
        tm = os.path.join(base_dir, f"{y}_team_matches.csv")
        pm = os.path.join(base_dir, f"{y}_player_matches.csv")
        if os.path.exists(tm) and os.path.exists(pm):
            years.append(y)
    return sorted(set(years))


def _parse_seasons_arg(seasons_arg: str, base_dir: str) -> List[int]:
    """
    Accepts:
      - 'all' -> scan base_dir to find all seasons present
      - single year, e.g. '2025'
      - comma list, e.g. '2023,2024,2025'
      - range, e.g. '2020-2025'
    """
    seasons_arg = seasons_arg.strip().lower()
    if seasons_arg == "all":
        det = _detect_seasons(base_dir)
        if not det:
            raise ValueError(
                f"No seasons found in '{base_dir}'. Expecting files like '2025_matches.csv' with companions."
            )
        return det

    rng = re.match(r"^(\d{4})\s*-\s*(\d{4})$", seasons_arg)
    if rng:
        start, end = int(rng.group(1)), int(rng.group(2))
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))

    parts = [p.strip() for p in seasons_arg.split(",") if p.strip()]
    out = []
    for p in parts:
        if not re.match(r"^\d{4}$", p):
            raise ValueError(
                f"Invalid season '{p}'. Use YYYY, CSV list, range 'YYYY-YYYY', or 'all'."
            )
        out.append(int(p))
    return sorted(set(out))


# ====================================================
# Core builder over ALL seasons (chronological)
# ====================================================
def build_features_all(
    base_dir: str,
    seasons: Iterable[int],
    output_csv: str,
    win_size_form: int = 5,
    win_size_home_away: int = 5,
    min_matches_for_factors: int = 2,
) -> pd.DataFrame:
    # 1) Load and normalize all seasons, then concatenate chronologically
    all_matches, all_team_matches, all_player_matches = [], [], []
    for season in seasons:
        print(f" Loading season {season}...")
        m_path = os.path.join(base_dir, f"{season}_matches.csv")
        tm_path = os.path.join(base_dir, f"{season}_team_matches.csv")
        pm_path = os.path.join(base_dir, f"{season}_player_matches.csv")

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
            columns=["home_team_id", "away_team_id"], errors="ignore"
        )

    # ---------- Attach match dates for sorting team_matches chronologically ----------
    if "date_utc" in team_matches.columns:
        team_matches["date_utc"] = pd.to_datetime(
            team_matches["date_utc"], errors="coerce"
        )
    else:
        team_matches = team_matches.merge(
            matches[["fixture_id", "date_utc"]], on="fixture_id", how="left"
        )

    # Sort by date (keep for xG calc), but do NOT drop date_utc yet—we need it
    team_matches = team_matches.sort_values("date_utc")

    # ---------- CALCULATE expected_goals from goals ----------
    team_matches = add_calculated_expected_goals(
        team_matches=team_matches,
        matches=matches,
        win_size_home_away=win_size_home_away,
        min_matches_for_factors=min_matches_for_factors,
    )

    # After calculation, we can drop helper date; we'll re-merge fresh later
    team_matches = team_matches.drop(columns=["date_utc"], errors="ignore")

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

    # 2.3 Form (rolling last N), shifted to avoid leakage
    w_form = win_size_form
    tm["form_points_lastN"] = (
        tm.groupby("team_id")["points"]
        .rolling(w_form, min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    # 2.4 Rolling xG/xGA (home/away) with transform, shifted
    w = win_size_home_away
    expg, oexpg, tid = tm["expected_goals"], tm["opp_expected_goals"], tm["team_id"]

    tm["is_home"] = tm["is_home"].astype(bool)
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

    # 2.6 TEAM RATING
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
            raise KeyError("Internal: tm lost 'team_id' before ratings merge.")
        tm = tm.merge(per_match_team_rating, on=["fixture_id", "team_id"], how="left")
        tm["team_rating"] = (
            tm.groupby("team_id")["team_avg_player_rating"]
            .rolling(win_size_form, min_periods=1)
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

    # Explicit pre-match xG for each side (from computed expected_goals)
    feat["pre_xG_home"] = home_tm["expected_goals"]
    feat["pre_xG_away"] = away_tm["expected_goals"]

    # Rolling xG/xGA features
    feat["home_xG_avg"] = home_tm["home_xG_avg_team"]
    feat["away_xG_avg"] = away_tm["away_xG_avg_team"]
    feat["home_xGA_avg"] = home_tm["home_xGA_avg_team"]
    feat["away_xGA_avg"] = away_tm["away_xGA_avg_team"]

    # Other features
    feat["home_team_rating"] = home_tm["team_rating"]
    feat["away_team_rating"] = away_tm["team_rating"]
    feat["home_form_points"] = home_tm["form_points_lastN"]
    feat["away_form_points"] = away_tm["form_points_lastN"]
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

    # 2.10 Key players missing (top-5 by minutes + 10*avg_rating), accumulated BEFORE each match
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
        # update AFTER match
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
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    feat = feat.sort_values(["date_utc", "fixture_id"])
    feat.to_csv(output_csv, index=False)

    print(f" Combined features saved to: {output_csv}")
    print(f"Rows: {len(feat):,} | Seasons: {sorted(matches['season'].unique())}")
    return feat


# ====================================================
# CLI
# ====================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-season feature builder.")
    p.add_argument(
        "--base-dir",
        default="data/csv_datasets/all",
        help="Folder with season CSVs (YYYY_matches.csv, YYYY_team_matches.csv, YYYY_player_matches.csv).",
    )
    p.add_argument(
        "--seasons",
        default="2025",
        help="Choose seasons: 'all', 'YYYY', 'YYYY,YYYY,YYYY', or 'YYYY-YYYY'. Example: --seasons 2023-2025",
    )
    p.add_argument(
        "--output",
        default="data/output/features_all_seasons.csv",
        help="Path to the single combined output CSV.",
    )
    p.add_argument(
        "--win-form",
        type=int,
        default=5,
        help="Rolling window for form & team ratings (default: 5).",
    )
    p.add_argument(
        "--win-homeaway",
        type=int,
        default=5,
        help="Rolling window for home/away xG & xGA (default: 5).",
    )
    p.add_argument(
        "--min-matches-factors",
        type=int,
        default=2,
        help="Min past matches at the venue before using attack/defense factors (default: 2).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seasons = _parse_seasons_arg(args.seasons, args.base_dir)
    print(f"Using seasons: {seasons}")
    build_features_all(
        base_dir=args.base_dir,
        seasons=seasons,
        output_csv=args.output,
        win_size_form=args.win_form,
        win_size_home_away=args.win_homeaway,
        min_matches_for_factors=args.min_matches_factors,
    )


if __name__ == "__main__":
    main()
