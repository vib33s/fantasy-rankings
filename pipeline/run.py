"""Weekly job: build features -> evaluate -> retrain -> predict -> docs/rankings.json"""
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

import features as F
import model as M

OUT = Path(__file__).resolve().parent.parent / "docs" / "rankings.json"


def injury_flags(season, week):
    """Map player_id -> report status for the target week, if the NFL has posted it."""
    try:
        inj = nfl.load_injuries([season]).to_pandas()
    except Exception:
        return {}
    inj = inj[(inj.week == week) & inj.report_status.notna()]
    return dict(zip(inj.gsis_id, inj.report_status))


def main():
    sched = F.load_schedule()
    season, week = F.find_target_week(sched)
    print(f"Target: {season} week {week}")

    weekly = F.load_weekly(range(F.FIRST_SEASON, season + 1))
    df = F.build_features(weekly, sched, season, week)
    future = df[(df.season == season) & (df.week == week)].copy()
    history = df[df[F.TARGET].notna()]

    # last season that is fully complete = our held-out test season
    done = sched.groupby("season").result.apply(lambda r: r.notna().all())
    test_season = int(done[done].index.max())
    print(f"Evaluating on {test_season} (validation: {test_season - 1})")
    metrics, best_iters = M.evaluate(history, test_season)
    for pos, m in metrics["by_position"].items():
        print(pos, "MAE", {k: round(v, 2) for k, v in m["mae"].items()},
              "Spearman", {k: round(v, 3) for k, v in m["spearman"].items()})

    preds = M.fit_final_and_predict(history, future, best_iters)

    # Only rank players with meaningful recent usage; hides practice-squad noise
    preds = preds[preds[f"{F.TARGET}_r5"].fillna(0) >= 3.0]
    flags = injury_flags(season, week)
    preds["injury"] = preds.player_id.map(flags)
    preds = preds[preds.injury != "Out"]
    preds["rank_overall"] = preds.proj.rank(ascending=False, method="first").astype(int)
    preds["rank_pos"] = preds.groupby("position").proj.rank(ascending=False, method="first").astype(int)
    preds = preds.sort_values("rank_overall")

    players = [{
        "name": r.player_display_name, "pos": r.position, "team": r.team,
        "opp": r.opponent_team, "home": bool(r.is_home) if pd.notna(r.is_home) else None,
        "proj": round(r.proj, 1), "floor": round(r.floor, 1), "ceiling": round(r.ceiling, 1),
        "last3": None if pd.isna(r.fantasy_points_ppr_r3) else round(r.fantasy_points_ppr_r3, 1),
        "implied": None if pd.isna(r.implied_total) else round(r.implied_total, 1),
        "injury": r.injury if isinstance(r.injury, str) else None,
        "rank": int(r.rank_overall), "pos_rank": int(r.rank_pos),
    } for r in preds.itertuples()]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "season": season, "week": week, "scoring": "PPR",
        "injuries_checked": bool(flags),
        "metrics": metrics, "players": players,
    }, indent=1))
    print(f"Wrote {len(players)} players to {OUT}")


if __name__ == "__main__":
    main()
