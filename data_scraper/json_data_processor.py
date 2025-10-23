"""
Reads per-fixture raw JSON (PL + cups/Europe + Championship) and produces per-season CSVs:

Output folder: ./data/csv_datasets/all

  - {season}_matches.csv          : one row per match
  - {season}_team_matches.csv     : one row per team per match
  - {season}_events.csv           : one row per timeline event (goals, cards, subs, VAR, etc.)
  - {season}_player_matches.csv   : one row per player per match with rich stats

Notes:
- The scraper attaches 'statistics', 'players', 'events', and 'lineups' directly on each fixture JSON.
- This processor is robust to missing sections (it will just skip those rows).
"""

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import ujson as json


RAW_DIR = "./data/raw_datasets/all"
OUT_DIR = "./data/csv_datasets/all"

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def _safe(d: Any, *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _as_dt(s: str):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# -----------------------------
# Parsers
# -----------------------------
def _parse_fixture_row(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one fixture to a match-level row."""
    fixture = doc.get("fixture", {})
    league = doc.get("league", {})
    teams = doc.get("teams", {})
    goals = doc.get("goals", {})
    score = doc.get("score", {})

    date_dt = _as_dt(_safe(fixture, "date", default=""))

    return {
        "fixture_id": _safe(fixture, "id"),
        "date_utc": date_dt.strftime("%Y-%m-%d %H:%M:%S") if date_dt else None,
        "season": _safe(league, "season"),
        "round": _safe(league, "round"),
        "league_id": _safe(league, "id"),
        "league_name": _safe(league, "name"),
        "country": _safe(league, "country"),
        "home_team_id": _safe(teams, "home", "id"),
        "home_team": _safe(teams, "home", "name"),
        "away_team_id": _safe(teams, "away", "id"),
        "away_team": _safe(teams, "away", "name"),
        "home_goals": _safe(goals, "home"),
        "away_goals": _safe(goals, "away"),
        "ht_home": _safe(score, "halftime", "home"),
        "ht_away": _safe(score, "halftime", "away"),
        "ft_home": _safe(score, "fulltime", "home"),
        "ft_away": _safe(score, "fulltime", "away"),
        "et_home": _safe(score, "extratime", "home"),
        "et_away": _safe(score, "extratime", "away"),
        "pen_home": _safe(score, "penalty", "home"),
        "pen_away": _safe(score, "penalty", "away"),
        "status": _safe(fixture, "status", "short"),
        "venue_id": _safe(fixture, "venue", "id"),
        "venue_name": _safe(fixture, "venue", "name"),
        "referee": _safe(fixture, "referee"),
    }


def _explode_team_rows(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Create team-centric rows (one row per team per match)."""
    home = matches_df.copy()
    home["team_id"] = home["home_team_id"]
    home["team_name"] = home["home_team"]
    home["opponent_id"] = home["away_team_id"]
    home["opponent"] = home["away_team"]
    home["goals_for"] = home["home_goals"]
    home["goals_against"] = home["away_goals"]
    home["is_home"] = True

    away = matches_df.copy()
    away["team_id"] = away["away_team_id"]
    away["team_name"] = away["away_team"]
    away["opponent_id"] = away["home_team_id"]
    away["opponent"] = away["home_team"]
    away["goals_for"] = away["away_goals"]
    away["goals_against"] = away["home_goals"]
    away["is_home"] = False

    cols = [
        "fixture_id",
        "date_utc",
        "season",
        "round",
        "league_id",
        "league_name",
        "country",
        "team_id",
        "team_name",
        "opponent_id",
        "opponent",
        "goals_for",
        "goals_against",
        "is_home",
    ]
    return pd.concat([home[cols], away[cols]], ignore_index=True)


def _parse_events_rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return one row per timeline event."""
    fixture = doc.get("fixture", {})
    league = doc.get("league", {})
    teams = doc.get("teams", {})
    events = doc.get("events", []) or []

    date_dt = _as_dt(_safe(fixture, "date", default=""))
    common = {
        "fixture_id": _safe(fixture, "id"),
        "date_utc": date_dt.strftime("%Y-%m-%d %H:%M:%S") if date_dt else None,
        "season": _safe(league, "season"),
        "league_id": _safe(league, "id"),
        "league_name": _safe(league, "name"),
        "home_team_id": _safe(teams, "home", "id"),
        "home_team": _safe(teams, "home", "name"),
        "away_team_id": _safe(teams, "away", "id"),
        "away_team": _safe(teams, "away", "name"),
    }

    rows = []
    for ev in events:
        row = dict(common)
        row.update(
            {
                "elapsed": _safe(ev, "time", "elapsed"),
                "elapsed_plus": _safe(ev, "time", "extra"),
                "team_id": _safe(ev, "team", "id"),
                "team_name": _safe(ev, "team", "name"),
                "player_id": _safe(ev, "player", "id"),
                "player_name": _safe(ev, "player", "name"),
                "assist_id": _safe(ev, "assist", "id"),
                "assist_name": _safe(ev, "assist", "name"),
                "type": _safe(ev, "type"),
                "detail": _safe(ev, "detail"),
                "comments": _safe(ev, "comments"),
            }
        )
        rows.append(row)
    return rows


def _parse_player_rows(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return one row per player per match from 'players' section.
    Structure is:
      players: [
        {
          "team": {...},
          "players": [
            {
              "player": {...},
              "statistics": [ { ... many sub-objects ... } ]
            },
            ...
          ]
        },
        ...
      ]
    We flatten the most common stats fields into top-level columns.
    """
    fixture = doc.get("fixture", {})
    league = doc.get("league", {})
    teams = doc.get("teams", {})
    blocks = doc.get("players", []) or []

    date_dt = _as_dt(_safe(fixture, "date", default=""))
    base = {
        "fixture_id": _safe(fixture, "id"),
        "date_utc": date_dt.strftime("%Y-%m-%d %H:%M:%S") if date_dt else None,
        "season": _safe(league, "season"),
        "league_id": _safe(league, "id"),
        "league_name": _safe(league, "name"),
        "home_team_id": _safe(teams, "home", "id"),
        "home_team": _safe(teams, "home", "name"),
        "away_team_id": _safe(teams, "away", "id"),
        "away_team": _safe(teams, "away", "name"),
    }

    out_rows: List[Dict[str, Any]] = []

    for block in blocks:
        team_id = _safe(block, "team", "id")
        team_name = _safe(block, "team", "name")

        for p in block.get("players", []) or []:
            player_id = _safe(p, "player", "id")
            player_name = _safe(p, "player", "name")

            # statistics is usually a list with 1 item per competition context
            for st in p.get("statistics", []) or []:
                games = _safe(st, "games", default={}) or {}
                shots = _safe(st, "shots", default={}) or {}
                goals = _safe(st, "goals", default={}) or {}
                passes = _safe(st, "passes", default={}) or {}
                tackles = _safe(st, "tackles", default={}) or {}
                duels = _safe(st, "duels", default={}) or {}
                dribbles = _safe(st, "dribbles", default={}) or {}
                fouls = _safe(st, "fouls", default={}) or {}
                cards = _safe(st, "cards", default={}) or {}
                penalty = _safe(st, "penalty", default={}) or {}

                row = dict(base)
                row.update(
                    {
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
                        "player_name": player_name,
                        # Games / role
                        "position": games.get("position"),
                        "rating": games.get("rating"),
                        "minutes": games.get("minutes"),
                        "captain": games.get("captain"),
                        "substitute": games.get("substitute"),
                        "number": games.get("number"),
                        "appearences": games.get("appearences"),  # API spelling
                        # Shots / Goals
                        "shots_total": shots.get("total"),
                        "shots_on": shots.get("on"),
                        "goals_total": goals.get("total"),
                        "goals_conceded": goals.get("conceded"),
                        "goals_assists": goals.get("assists"),
                        "goals_saves": goals.get("saves"),
                        # Passes
                        "passes_total": passes.get("total"),
                        "passes_key": passes.get("key"),
                        "passes_accuracy": passes.get("accuracy"),
                        # Defensive
                        "tackles_total": tackles.get("total"),
                        "tackles_blocks": tackles.get("blocks"),
                        "tackles_interceptions": tackles.get("interceptions"),
                        "duels_total": duels.get("total"),
                        "duels_won": duels.get("won"),
                        "dribbles_attempts": dribbles.get("attempts"),
                        "dribbles_success": dribbles.get("success"),
                        "dribbles_past": dribbles.get("past"),
                        "fouls_drawn": fouls.get("drawn"),
                        "fouls_committed": fouls.get("committed"),
                        "cards_yellow": cards.get("yellow"),
                        "cards_red": cards.get("red"),
                        "penalty_won": penalty.get("won"),
                        "penalty_committed": penalty.get("committed"),
                        "penalty_scored": penalty.get("scored"),
                        "penalty_missed": penalty.get("missed"),
                        "penalty_saved": penalty.get("saved"),
                    }
                )
                out_rows.append(row)

    return out_rows


# -----------------------------
# Main
# -----------------------------
def process_all():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    if not files:
        print(f"No raw JSON found in {RAW_DIR}")
        return

    match_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    player_rows: List[Dict[str, Any]] = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        # Matches
        try:
            match_rows.append(_parse_fixture_row(doc))
        except Exception as e:
            print(f"[WARN] Failed to parse match row {fp}: {e}")

        # Events
        try:
            event_rows.extend(_parse_events_rows(doc))
        except Exception as e:
            print(f"[WARN] Failed to parse events in {fp}: {e}")

        # Player stats
        try:
            player_rows.extend(_parse_player_rows(doc))
        except Exception as e:
            print(f"[WARN] Failed to parse players in {fp}: {e}")

    # Build dataframes
    matches_df = pd.DataFrame(match_rows).dropna(
        subset=["fixture_id", "date_utc", "home_team_id", "away_team_id"]
    )
    events_df = pd.DataFrame(event_rows)
    players_df = pd.DataFrame(player_rows)

    # Write per-season files
    for season, grp in matches_df.groupby("season"):
        try:
            season = int(season)
        except Exception:
            print(f"[WARN] Bad season value {season}, skipping...")
            continue

        # Matches
        out_matches = Path(OUT_DIR) / f"{season}_matches.csv"
        out_team_matches = Path(OUT_DIR) / f"{season}_team_matches.csv"
        grp_sorted = grp.sort_values(["date_utc", "fixture_id"])
        grp_sorted.to_csv(out_matches, index=False)

        team_df = _explode_team_rows(grp_sorted)
        team_df = team_df.sort_values(["team_id", "date_utc"])
        team_df.to_csv(out_team_matches, index=False)
        print(f"Written: {out_matches}")
        print(f"Written: {out_team_matches}")

        # Events
        if not events_df.empty:
            ev_grp = events_df.loc[events_df["season"] == season].copy()
            if not ev_grp.empty:
                ev_grp = ev_grp.sort_values(
                    ["fixture_id", "elapsed", "elapsed_plus"]
                ).reset_index(drop=True)
                out_events = Path(OUT_DIR) / f"{season}_events.csv"
                ev_grp.to_csv(out_events, index=False)
                print(f"Written: {out_events}")

        # Player matches
        if not players_df.empty:
            pl_grp = players_df.loc[players_df["season"] == season].copy()
            if not pl_grp.empty:
                pl_grp = pl_grp.sort_values(
                    ["fixture_id", "team_id", "player_id"]
                ).reset_index(drop=True)
                out_players = Path(OUT_DIR) / f"{season}_player_matches.csv"
                pl_grp.to_csv(out_players, index=False)
                print(f"Written: {out_players}")


if __name__ == "__main__":
    process_all()
