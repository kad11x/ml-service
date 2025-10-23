import requests
import ujson as json
import os


RAW_DIR = "./data/raw_datasets/epl"


class Scraper:
    """
    Scrapes a Premier League season from API-SPORTS by FIXTURE ID and
    writes ONE RAW JSON FILE PER FIXTURE to ./data/raw_datasets/epl/.

    Each saved JSON contains the core fixture plus:
      - statistics  (team stats for the match)
      - players     (per-player stats for the match)
      - events      (timeline)
      - lineups     (formations, startXI)
    The transformation into 4 CSVs is handled by json_data_processor.py.
    """

    def __init__(self, year: str):
        self.year = str(year)
        # keep your header style (do not hardcode your key here)
<<<<<<< HEAD
        self.headers = {"X-RapidAPI-Key": "872ef3873119ef07b047c12db204ab2c"}
        self.league = "39"  # EPL
=======
        self.headers = {"X-RapidAPI-Key": "***"}
        self.league = "39"  # EPL # add championship for the teams that have been added to premier league and for the rest days to be complete, we need to add international break, turnements and so on: 
        # international break, FA cup, Carabao cup, comunity shield, championse legue, europa and confrance.  
>>>>>>> 0c919e9840d28cd862c5979fdd58e4b0ffcc2dd9

        os.makedirs(RAW_DIR, exist_ok=True)

    # ---------------------------
    # Public
    # ---------------------------
    def scrape(self):
        """Fetch season fixtures, then fetch each fixture by ID and save combined raw JSON per fixture."""
        season_fixtures = self._fetch_season_fixtures()
        if not season_fixtures:
            print("No fixtures returned for season", self.year)
            return

        fixture_ids = [
            fixt["fixture"]["id"] for fixt in season_fixtures if "fixture" in fixt
        ]
        print(f"Season {self.year}: {len(fixture_ids)} fixtures found.")

        for i, fid in enumerate(fixture_ids, 1):
            try:
                self._process_fixture(fid)
                if i % 20 == 0:
                    print(f"Saved {i}/{len(fixture_ids)} fixtures...")
            except Exception as e:
                print(f"[WARN] fixture {fid} failed: {e}")

        print(f"Done. Raw JSON written to {RAW_DIR}")

    # ---------------------------
    # API calls
    # ---------------------------
    def _fetch_season_fixtures(self):
        """
        Get ALL fixtures for a season (no status filter, to be comprehensive).
        You can add &status=FT if you only want finished matches.
        """
        url = (
            "https://v3.football.api-sports.io/fixtures"
            f"?league={self.league}&season={self.year}&status=FT"
        )
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])

    def _fetch_fixture_core(self, fixture_id):
        url = f"https://v3.football.api-sports.io/fixtures?id={fixture_id}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        res = r.json().get("response", [])
        return res[0] if res else None

    def _fetch_fixture_statistics(self, fixture_id):
        url = f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    def _fetch_fixture_players(self, fixture_id):
        url = f"https://v3.football.api-sports.io/fixtures/players?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    def _fetch_fixture_events(self, fixture_id):
        url = f"https://v3.football.api-sports.io/fixtures/events?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    def _fetch_fixture_lineups(self, fixture_id):
        url = f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}"
        r = requests.get(url, headers=self.headers)
        if r.status_code != 200:
            return []
        return r.json().get("response", [])

    # ---------------------------
    # Save per fixture
    # ---------------------------
    def _process_fixture(self, fixture_id):
        """
        Build one combined JSON object for the fixture:
          - Start from core (the usual fixture document)
          - Attach .statistics, .players, .events, .lineups
        Then save as {season}_{fixture_id}.json
        """
        core = self._fetch_fixture_core(fixture_id)
        if not core:
            raise RuntimeError("empty core")

        # attach extra sections so the processor can find them directly
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

    season_arg = sys.argv[1] if len(sys.argv) > 1 else "2025"
    Scraper(season_arg).scrape()
