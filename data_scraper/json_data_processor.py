import os
import ujson as json
import csv
from collections import defaultdict
from typing import Any, Dict, Iterable, List

RAW_DIR = "./data/raw_datasets/epl"
OUT_DIR = "./data/csv_datasets/epl"


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def normalize_fixtures(obj: Any) -> List[Dict]:
    # Accept:
    #  - single full fixture dict (your scraper output)
    #  - list of fixture dicts
    #  - {"response": [fixtures]} (API style)
    if obj is None:
        return []
    if isinstance(obj, dict):
        if isinstance(obj.get("response"), list):
            return obj["response"]
        if "fixture" in obj and "league" in obj:
            return [obj]
        vals = list(obj.values())
        if vals and isinstance(vals[0], dict) and "fixture" in vals[0]:
            return vals
        return []
    if isinstance(obj, list):
        return obj
    return []


def stats_list_to_cols(lst: Iterable[Dict]) -> Dict[str, Any]:
    out = {}
    for it in lst or []:
        key = str(it.get("type", "")).lower().replace(" ", "_")
        val = it.get("value")
        if isinstance(val, str) and val.endswith("%"):
            try:
                val = float(val.strip("%")) / 100.0
            except Exception:
                pass
        out[key] = val
    return out


def match_row_from_fixture(core: Dict) -> Dict[str, Any]:
    fixture = core.get("fixture", {}) or {}
    league = core.get("league", {}) or {}
    teams = core.get("teams", {}) or {}
    goals = core.get("goals", {}) or {}
    lineups = core.get("lineups", None)

    home_form = away_form = None
    if isinstance(lineups, list):
        if len(lineups) > 0:
            home_form = lineups[0].get("formation")
        if len(lineups) > 1:
            away_form = lineups[1].get("formation")

    return {
        "fixture_id": fixture.get("id"),
        "date_utc": fixture.get("date"),
        "timezone": fixture.get("timezone"),
        "status_short": (fixture.get("status") or {}).get("short"),
        "status_long": (fixture.get("status") or {}).get("long"),
        "season": league.get("season"),
        "round": league.get("round"),
        "league_id": league.get("id"),
        "league_name": league.get("name"),
        "venue_id": (fixture.get("venue") or {}).get("id"),
        "venue_name": (fixture.get("venue") or {}).get("name"),
        "venue_city": (fixture.get("venue") or {}).get("city"),
        "referee": fixture.get("referee"),
        "home_team_id": (teams.get("home") or {}).get("id"),
        "home_team": (teams.get("home") or {}).get("name"),
        "away_team_id": (teams.get("away") or {}).get("id"),
        "away_team": (teams.get("away") or {}).get("name"),
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "home_formation": home_form,
        "away_formation": away_form,
    }


def team_rows_from_fixture(core: Dict, ft_only: bool = True) -> List[Dict[str, Any]]:
    rows = []
    m = match_row_from_fixture(core)
    if ft_only and m.get("status_short") != "FT":
        return rows
    stats_blocks = core.get("statistics") or []
    for block in stats_blocks:
        team = block.get("team", {}) or {}
        stat_cols = stats_list_to_cols(block.get("statistics"))
        is_home = team.get("id") == m.get("home_team_id")
        gf = m.get("home_goals") if is_home else m.get("away_goals")
        ga = m.get("away_goals") if is_home else m.get("home_goals")
        result = (
            None
            if (gf is None or ga is None)
            else ("WIN" if gf > ga else ("DRAW" if gf == ga else "LOSS"))
        )
        rows.append(
            {
                "fixture_id": m.get("fixture_id"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "is_home": is_home,
                "goals_for": gf,
                "goals_against": ga,
                "result": result,
                **stat_cols,
            }
        )
    return rows


def player_rows_from_fixture(core: Dict, ft_only: bool = True) -> List[Dict[str, Any]]:
    out = []
    m = match_row_from_fixture(core)
    if ft_only and m.get("status_short") != "FT":
        return out
    for tpack in core.get("players") or []:
        tid = (tpack.get("team") or {}).get("id")
        tname = (tpack.get("team") or {}).get("name")
        for p in tpack.get("players") or []:
            pl = p.get("player") or {}
            st = (p.get("statistics") or [{}])[0]
            games = st.get("games") or {}
            shots = st.get("shots") or {}
            goals_s = st.get("goals") or {}
            passes = st.get("passes") or {}
            tackles = st.get("tackles") or {}
            duels = st.get("duels") or {}
            dribbles = st.get("dribbles") or {}
            fouls = st.get("fouls") or {}
            cards = st.get("cards") or {}
            try:
                rating = (
                    float(games.get("rating"))
                    if games.get("rating") not in [None, ""]
                    else None
                )
            except Exception:
                rating = None
            out.append(
                {
                    "fixture_id": m.get("fixture_id"),
                    "team_id": tid,
                    "team_name": tname,
                    "player_id": pl.get("id"),
                    "player_name": pl.get("name"),
                    "minutes": games.get("minutes"),
                    "number": games.get("number"),
                    "position": games.get("position"),
                    "rating": rating,
                    "captain": games.get("captain"),
                    "starter": not games.get("substitute", False),
                    "shots_total": shots.get("total"),
                    "shots_on": shots.get("on"),
                    "goals": goals_s.get("total"),
                    "assists": goals_s.get("assists"),
                    "passes_total": passes.get("total"),
                    "passes_key": passes.get("key"),
                    "passes_accuracy": passes.get("accuracy"),
                    "tackles_total": tackles.get("total"),
                    "blocks": tackles.get("blocks"),
                    "interceptions": tackles.get("interceptions"),
                    "duels_total": duels.get("total"),
                    "duels_won": duels.get("won"),
                    "dribbles_attempts": dribbles.get("attempts"),
                    "dribbles_success": dribbles.get("success"),
                    "fouls_drawn": fouls.get("drawn"),
                    "fouls_committed": fouls.get("committed"),
                    "yellow": cards.get("yellow"),
                    "red": cards.get("red"),
                }
            )
    return out


def event_rows_from_fixture(core: Dict, ft_only: bool = True) -> List[Dict[str, Any]]:
    rows = []
    m = match_row_from_fixture(core)
    if ft_only and m.get("status_short") != "FT":
        return rows
    for e in core.get("events") or []:
        time = e.get("time") or {}
        team = e.get("team") or {}
        player = e.get("player") or {}
        assist = e.get("assist") or {}
        rows.append(
            {
                "fixture_id": m.get("fixture_id"),
                "minute": time.get("elapsed"),
                "minute_extra": time.get("extra"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "type": e.get("type"),
                "detail": e.get("detail"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "assist_id": assist.get("id"),
                "assist_name": assist.get("name"),
            }
        )
    return rows


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def output_data():
    ensure_dirs()
    per_season = defaultdict(
        lambda: {
            "match_rows": [],
            "team_rows": [],
            "player_rows": [],
            "event_rows": [],
        }
    )

    files = [fn for fn in os.listdir(RAW_DIR) if fn.endswith(".json")]
    files.sort()  # stable order

    for fn in files:
        path = os.path.join(RAW_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] could not read {fn}: {e}")
            continue

        fixtures = normalize_fixtures(data)
        if not fixtures:
            # This can happen if the file is an empty {} etc.
            continue

        for core in fixtures:
            m = match_row_from_fixture(core)
            season = m.get("season") or "unknown"
            buf = per_season[season]
            buf["match_rows"].append(m)
            buf["team_rows"].extend(team_rows_from_fixture(core, ft_only=True))
            buf["player_rows"].extend(player_rows_from_fixture(core, ft_only=True))
            buf["event_rows"].extend(event_rows_from_fixture(core, ft_only=True))

    # write once per season (no overwriting per file!)
    for season, payload in per_season.items():
        matches_csv = os.path.join(OUT_DIR, f"{season}_matches.csv")
        team_csv = os.path.join(OUT_DIR, f"{season}_team_matches.csv")
        player_csv = os.path.join(OUT_DIR, f"{season}_player_matches.csv")
        events_csv = os.path.join(OUT_DIR, f"{season}_events.csv")

        match_cols = [
            "fixture_id",
            "date_utc",
            "timezone",
            "status_short",
            "status_long",
            "season",
            "round",
            "league_id",
            "league_name",
            "venue_id",
            "venue_name",
            "venue_city",
            "referee",
            "home_team_id",
            "home_team",
            "away_team_id",
            "away_team",
            "home_goals",
            "away_goals",
            "home_formation",
            "away_formation",
        ]

        team_base = [
            "fixture_id",
            "team_id",
            "team_name",
            "is_home",
            "goals_for",
            "goals_against",
            "result",
        ]
        dyn_keys = set()
        for r in payload["team_rows"]:
            for k in r.keys():
                if k not in team_base:
                    dyn_keys.add(k)
        team_cols = team_base + sorted(dyn_keys)

        player_cols = [
            "fixture_id",
            "team_id",
            "team_name",
            "player_id",
            "player_name",
            "minutes",
            "number",
            "position",
            "rating",
            "captain",
            "starter",
            "shots_total",
            "shots_on",
            "goals",
            "assists",
            "passes_total",
            "passes_key",
            "passes_accuracy",
            "tackles_total",
            "blocks",
            "interceptions",
            "duels_total",
            "duels_won",
            "dribbles_attempts",
            "dribbles_success",
            "fouls_drawn",
            "fouls_committed",
            "yellow",
            "red",
        ]
        event_cols = [
            "fixture_id",
            "minute",
            "minute_extra",
            "team_id",
            "team_name",
            "type",
            "detail",
            "player_id",
            "player_name",
            "assist_id",
            "assist_name",
        ]

        write_csv(matches_csv, match_cols, payload["match_rows"])
        write_csv(team_csv, team_cols, payload["team_rows"])
        write_csv(player_csv, player_cols, payload["player_rows"])
        write_csv(events_csv, event_cols, payload["event_rows"])


if __name__ == "__main__":
    output_data()
