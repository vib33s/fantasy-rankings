"""Load nflverse data and build model features.

The one rule that matters: every feature for a game is computed from games
BEFORE it. We enforce that with .shift(1) before every rolling/expanding
calculation, and by appending the upcoming week as rows with empty stats so
the exact same code builds features for training rows and prediction rows.
"""
import numpy as np
import pandas as pd
import nflreadpy as nfl

POSITIONS = ["QB", "RB", "WR", "TE"]
FIRST_SEASON = 2018
TARGET = "fantasy_points_ppr"  # scoring format: full PPR

ROLL_STATS = [
    "fantasy_points_ppr", "targets", "carries", "target_share",
    "wopr", "receiving_air_yards", "attempts", "passing_epa",
]


def load_schedule(first=FIRST_SEASON):
    last = nfl.get_current_season()
    s = nfl.load_schedules(list(range(first, last + 1))).to_pandas()
    return s[s.game_type == "REG"].copy()


def load_weekly(seasons):
    frames = []
    for season in seasons:
        try:
            frames.append(nfl.load_player_stats([season]).to_pandas())
        except Exception:
            pass  # season has no data yet (e.g. preseason)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.season_type == "REG") & df.position.isin(POSITIONS)]
    keep = ["player_id", "player_display_name", "position", "season", "week",
            "team", "opponent_team"] + ROLL_STATS
    return df[keep].drop_duplicates(["player_id", "season", "week"]).copy()


def find_target_week(sched):
    """The first week where no game has been played yet.

    (Using "first unplayed game" would pick a week that's still in progress,
    e.g. on a Monday before the Monday-night game.)"""
    played = sched.groupby(["season", "week"]).result.apply(lambda r: r.notna().any())
    todo = played[~played]
    if todo.empty:
        raise SystemExit("No upcoming weeks found - the season is over.")
    season, week = todo.index.min()
    return int(season), int(week)


def team_games(sched):
    """One row per team per game, with Vegas-derived context."""
    home = pd.DataFrame({
        "season": sched.season, "week": sched.week, "team": sched.home_team,
        "opponent": sched.away_team, "is_home": 1,
        "team_spread": sched.spread_line,  # positive = favored
        "implied_total": (sched.total_line + sched.spread_line) / 2,
        "rest": sched.home_rest,
    })
    away = pd.DataFrame({
        "season": sched.season, "week": sched.week, "team": sched.away_team,
        "opponent": sched.home_team, "is_home": 0,
        "team_spread": -sched.spread_line,
        "implied_total": (sched.total_line - sched.spread_line) / 2,
        "rest": sched.away_rest,
    })
    return pd.concat([home, away], ignore_index=True)


def defense_vs_position(weekly, tg):
    """Rolling average of fantasy points each defense allowed to each position
    over its previous 5 games (shifted, so it excludes the game itself)."""
    allowed = (weekly.groupby(["season", "week", "opponent_team", "position"])
               [TARGET].sum().rename("allowed").reset_index()
               .rename(columns={"opponent_team": "team"}))
    grid = tg[["season", "week", "team"]].merge(
        pd.DataFrame({"position": POSITIONS}), how="cross")
    grid = grid.merge(allowed, how="left", on=["season", "week", "team", "position"])
    grid = grid.sort_values(["team", "position", "season", "week"])
    grid["opp_allowed_roll5"] = (grid.groupby(["team", "position"]).allowed
                                 .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean()))
    return grid.rename(columns={"team": "opponent"})[
        ["season", "week", "opponent", "position", "opp_allowed_roll5"]]


def upcoming_rows(weekly, tg, season, week):
    """Rows (with empty stats) for players we want to rank in the target week:
    anyone who played in the last 4 completed weeks, on a team not on bye."""
    ordinal = weekly.season * 100 + weekly.week
    target_ord = season * 100 + week
    recent_weeks = sorted(ordinal[ordinal < target_ord].unique())[-4:]
    recent = weekly[ordinal.isin(recent_weeks)]
    latest = (recent.sort_values(["season", "week"])
              .groupby("player_id").tail(1)
              [["player_id", "player_display_name", "position", "team"]])
    games = tg[(tg.season == season) & (tg.week == week)][["team", "opponent"]]
    out = latest.merge(games, on="team", how="inner")  # inner join drops bye teams
    out["season"], out["week"] = season, week
    out["opponent_team"] = out.opponent
    return out.drop(columns="opponent").drop_duplicates("player_id")


def build_features(weekly, sched, season, week):
    tg = team_games(sched)
    dvp = defense_vs_position(weekly, tg)

    # history + upcoming week (stats empty) in one frame
    df = pd.concat([weekly, upcoming_rows(weekly, tg, season, week)], ignore_index=True)
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # --- rolling form: previous 3 and 5 games, never including the current one
    g = df.groupby("player_id")
    for stat in ROLL_STATS:
        df[f"{stat}_r3"] = g[stat].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df[f"{stat}_r5"] = g[stat].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["ppr_std5"] = g[TARGET].transform(lambda x: x.shift(1).rolling(5, min_periods=3).std())
    df["season_to_date"] = (df.groupby(["player_id", "season"])[TARGET]
                            .transform(lambda x: x.shift(1).expanding().mean()))
    df["career_games"] = g.cumcount()

    # --- last season's average points per game
    prev = (weekly.groupby(["player_id", "season"])[TARGET].mean()
            .rename("prev_season_avg").reset_index())
    prev["season"] += 1
    df = df.merge(prev, on=["player_id", "season"], how="left")

    # --- game context: Vegas lines, home/away, rest, opponent defense
    ctx = tg.rename(columns={"opponent": "opp_sched"})
    df = df.merge(ctx, on=["season", "week", "team"], how="left")
    df = df.merge(dvp, left_on=["season", "week", "opponent_team", "position"],
                  right_on=["season", "week", "opponent", "position"], how="left")
    df["pos_code"] = df.position.map({p: i for i, p in enumerate(POSITIONS)})

    return df


FEATURES = (
    [f"{s}_r3" for s in ROLL_STATS] + [f"{s}_r5" for s in ROLL_STATS] +
    ["ppr_std5", "season_to_date", "prev_season_avg", "career_games",
     "is_home", "team_spread", "implied_total", "rest", "opp_allowed_roll5", "week"]
)
