import os
from typing import Iterable, List, Union

import requests
import ujson as json

# Store all raw JSON here (shared by all competitions)
RAW_DIR = "./data/raw_datasets/all"


class Scraper:
    """
    Scrape fixtures for one season across multiple competitions, but only save
    non-PL fixtures that involve at least one Premier League (league 39) team.

    For each saved fixture, we also attach:
      - statistics
      - players
      - events
      - lineups

    Output: one JSON per fixture in RAW_DIR: {season}_{fixture_id}.json
    """

    # API-FOOTBALL league IDs (adjust if your account differs)
    DEFAULT_LEAGUES = [
        39,  # Premier League
        40,  # Championship
        45,  # FA Cup
        48,  # EFL Cup (Carabao)
        528,  # Community Shield
        2,  # UEFA Champions League
        3,  # UEFA Europa League
        848,  # UEFA Europa Conference League
    ]

    def __init__(
        self, year: Union[int, str], leagues: Union[None, Iterable[int]] = None
    ):
        self.year = str(year)

        # ✅ Hardcoded API key (back to original style)
        self.headers = {"X-RapidAPI-Key": "872ef3873119ef07b047c12db204ab2c"}

        # Allow override of leagues via param or env var "FOOTBALL_LEAGUE_IDS=39,45,..."
        env_leagues = os.getenv("FOOTBALL_LEAGUE_IDS")
        if leagues is None and env_leagues:
            try:
                leagues = [int(x.strip()) for x in env_leagues.split(",") if x.strip()]
            except Exception:
                leagues = None

        self.leagues: List[int] = (
            list(leagues) if leagues is not None else list(self.DEFAULT_LEAGUES)
        )

        os.makedirs(RAW_DIR, exist_ok=True)

    # ---------------------------
    # Helpers
    # ---------------------------
    def _get_pl_team_ids(self) -> set:
        """Return the set of team IDs playing Premier League (league 39) this season."""
        try:
            url = f"https://v3.football.api-sports.io/fixtures?league=39&season={self.year}&status=FT"
            r = requests.get(url, headers=self.headers)
            r.raise_for_status()
            resp = r.json().get("response", [])
        except Exception:
            resp = []

        ids = set()
        for fixt in resp:
            try:
                th = fixt["teams"]["home"]["id"]
                ta = fixt["teams"]["away"]["id"]
                if th:
                    ids.add(th)
                if ta:
                    ids.add(ta)
            except Exception:
                continue
        return ids

    # ---------------------------
    # Public
    # ---------------------------
    def scrape(self):
        """
        Fetch fixtures for the configured competitions. For non-PL leagues we keep
        ONLY fixtures where home or away is a PL team for this season.
        """
        pl_team_ids = self._get_pl_team_ids()
        if not pl_team_ids:
            print(
                "WARNING: Could not resolve PL team IDs from league 39 (status=FT). Proceeding without filtering."
            )
        all_fixture_ids = []

        for lg in self.leagues:
            season_fixtures = self._fetch_season_fixtures(lg)
            if not season_fixtures:
                print(f"No fixtures returned for league {lg} season {self.year}")
                continue

            if lg == 39 or not pl_team_ids:
                filtered = season_fixtures
            else:
                # keep only if at least one PL team is involved
                filtered = []
                for fixt in season_fixtures:
                    try:
                        th = fixt["teams"]["home"]["id"]
                        ta = fixt["teams"]["away"]["id"]
                        if (th in pl_team_ids) or (ta in pl_team_ids):
                            filtered.append(fixt)
                    except Exception:
                        continue

            fixture_ids = [
                fixt["fixture"]["id"] for fixt in filtered if "fixture" in fixt
            ]
            print(
                f"League {lg} season {self.year}: {len(fixture_ids)} fixtures kept "
                f"(filtered from {len(season_fixtures)})."
            )
            all_fixture_ids.extend(fixture_ids)

        # Deduplicate across competitions
        all_fixture_ids = sorted(set(all_fixture_ids))
        print(f"Total fixtures to process across leagues: {len(all_fixture_ids)}")

        for i, fid in enumerate(all_fixture_ids, 1):
            try:
                self._process_fixture(fid)
                if i % 20 == 0:
                    print(f"Saved {i}/{len(all_fixture_ids)} fixtures...")
            except Exception as e:
                print(f"[WARN] fixture {fid} failed: {e}")

        print(f"Done. Raw JSON written to {RAW_DIR}")

    # ---------------------------
    # API calls
    # ---------------------------
    def _fetch_season_fixtures(self, league_id: int):
        """
        Get ALL fixtures for a season for a league/tournament.
        Using status=FT to focus on finished matches with stable stats.
        """
        url = (
            "https://v3.football.api-sports.io/fixtures"
            f"?league={league_id}&season={self.year}&status=FT"
        )
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])

    def _fetch_fixture_core(self, fixture_id: int):
        url = f"https://v3.football.api-sports.io/fixtures?id={fixture_id}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        res = r.json().get("response", [])
        return res[0] if res else None

    def _fetch_fixture_statistics(self, fixture_id: int):
        url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    def _fetch_fixture_players(self, fixture_id: int):
        url = f"https://v3.football.api-sports.io/fixtures/players?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    def _fetch_fixture_events(self, fixture_id: int):
        url = f"https://v3.football.api-sports.io/fixtures/events?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    def _fetch_fixture_lineups(self, fixture_id: int):
        url = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    # ---------------------------
    # Save per fixture
    # ---------------------------
    def _process_fixture(self, fixture_id: int):
        """Attach extra sections and save one combined JSON per fixture."""
        core = self._fetch_fixture_core(fixture_id)
        if not core:
            raise RuntimeError("empty core")

        core["statistics"] = self._fetch_fixture_statistics(fixture_id)
        core["players"] = self._fetch_fixture_players(fixture_id)
        core["events"] = self._fetch_fixture_events(fixture_id)
        core["lineups"] = self._fetch_fixture_lineups(fixture_id)

        out_name = os.path.join(RAW_DIR, f"{self.year}_{fixture_id}.json")
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(core, f)
        return True


if __name__ == "__main__":
    import sys

    # Usage examples:
    #   python api_data_scraper.py 2025
    #   python api_data_scraper.py 2025 39,40,45,48,528,2,3,848
    season_arg = sys.argv[1] if len(sys.argv) > 1 else "2025"
    leagues_arg = None
    if len(sys.argv) > 2:
        try:
            leagues_arg = [int(x.strip()) for x in sys.argv[2].split(",") if x.strip()]
        except Exception:
            leagues_arg = None

    Scraper(season_arg, leagues=leagues_arg).scrape()
