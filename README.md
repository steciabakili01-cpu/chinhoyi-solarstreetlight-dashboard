# Chinhoyi Smart Solar Streetlight Management & Decision-Support Dashboard

Role-based municipal Web GIS dashboard built with Panel, GeoPandas, Bokeh,
Shapely, PyProj and SQLite.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open the local URL printed in the terminal (normally http://localhost:5006).

## Deploy to Render

This repository uses the Dockerfile. In Render, create a **Web Service**,
connect this GitHub repository, and select the **Docker** runtime if prompted.
Render will build the Dockerfile and run `python app.py`.

The application reads Render's `PORT` environment variable and its
`RENDER_EXTERNAL_HOSTNAME` to configure the Panel server and WebSocket origin.

## Required security environment variables

Set these in Render → your service → Environment. Use strong, unique values;
never commit real passwords or API keys to GitHub.

- `ADMIN_DEFAULT_PASSWORD`
- `STAFF_DEFAULT_PASSWORD`
- `COMMUNITY_DEFAULT_PASSWORD`

Optional:
- `ANTHROPIC_API_KEY` — enables the optional Anthropic-backed assistant.
- `ANTHROPIC_MODEL` — optional model override.
- `DB_PATH` — SQLite database path. If using a Render persistent disk, mount it
  (for example at `/var/data`) and set `DB_PATH=/var/data/app_data.db`.
- `PANEL_ALLOW_WS_ORIGIN` — optional explicit allowed hostname(s), comma-separated.
  Normally the app uses Render's `RENDER_EXTERNAL_HOSTNAME` automatically.
- `DEFAULT_BUDGET_USD`, `DEFAULT_UNIT_COST_USD`, `DEFAULT_CONTINGENCY_PCT`
- `DEFAULT_COVERAGE_RADIUS_M`, `DEFAULT_MAX_CANDIDATES`, `CANDIDATE_SPACING_M`
- `MIN_NEW_LIGHT_SPACING_M`, `SESSION_TTL_HOURS`, `LOG_LEVEL`

## Data and assets

Keep spatial source layers in `data/` and the approved municipal logo in
`assets/municipality_logo.png` (or a supported image format). Ensure all files
needed by the dashboard are committed to the repository, subject to GitHub's
file-size limits and municipal data permissions.

## Important deployment notes

- The application has development fallback credentials documented in its source.
  Override all three password environment variables before exposing the service.
- Render's local container filesystem is ephemeral unless a persistent disk is
  attached. SQLite user accounts, sessions, reports and other database changes
  can be lost when the service is restarted or redeployed without persistent storage.
- A persistent disk is suitable for a single service instance; do not scale this
  SQLite deployment to multiple instances.
- Do not upload confidential resident, crime, or municipal operational data to a
  public repository. Remove sensitive data and use access-controlled storage.
- Review basemap provider terms and licensing before public or operational rollout.
- Demo layers and simulated streetlight status must be clearly distinguished from
  verified municipal survey data.
