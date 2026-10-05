# Fantasy football rankings

Predicts next week's PPR fantasy points for QB/RB/WR/TE with one LightGBM model
per position, then publishes a static page.

## Run it locally

    pip install -r requirements.txt
    python pipeline/run.py          # writes docs/rankings.json
    cd docs && python -m http.server 8000
    # open http://localhost:8000

## How it works

- `pipeline/features.py`  loads nflverse data, builds rolling-form, Vegas, and opponent features
- `pipeline/model.py`     trains/evaluates the models, compares against baselines, predicts
- `pipeline/run.py`       runs everything and exports `docs/rankings.json`
- `docs/index.html`       the website (reads rankings.json, no build step)
- `.github/workflows/update.yml`  re-runs the pipeline on a schedule

## Publish

1. Push this folder to a GitHub repo.
2. Settings > Pages > Deploy from branch > `main` / `/docs`.
3. Actions tab > "Update rankings" > Run workflow once to test the schedule job.
