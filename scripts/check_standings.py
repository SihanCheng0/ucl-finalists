"""Compare the 10 finalists' computed group/league-phase points and goal difference with UEFA standings."""
import json
import urllib.request

from ucl import config, dataset
from ucl.uefa import USER_AGENT


def standings_item(season: int, team_id: str) -> dict | None:
    request = urllib.request.Request(config.STANDINGS_URL.format(season=season), headers={"User-Agent": USER_AGENT})
    groups = json.loads(urllib.request.urlopen(request, timeout=25).read())
    return next((item for g in groups for item in g["items"] if str(item["team"]["id"]) == team_id), None)


def main() -> int:
    ts = dataset.load().team_seasons
    bad = 0
    for r in ts.loc[ts["is_target"]].itertuples():
        label = f"{config.season_label(int(r.season))} {r.team_display}"
        item = standings_item(int(r.season), r.team_id)
        if item is None:
            print(f"BAD {label}: team id {r.team_id} not found in UEFA standings")
            bad += 1
            continue
        ours = (round(r.points_pg * r.n_matches), round(r.goal_diff_pg * r.n_matches))
        theirs = (int(item["points"]), int(item["goalDifference"]))
        bad += ours != theirs
        print(f"{'OK ' if ours == theirs else 'BAD'} {label}: ours {ours[0]} pts {ours[1]:+d} GD | "
              f"UEFA {theirs[0]} pts {theirs[1]:+d} GD")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
