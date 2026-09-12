FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Background refresh OFF — we rely on the on-request refresh (~2s on first
    # visit, throttled to 120s afterwards) so the container stays idle when
    # no one's browsing. Set STRIKECAST_DISABLE_REFRESH=0 to also enable the
    # Hourly background loop if you want predictions to update with no traffic.
    STRIKECAST_DISABLE_REFRESH=1 \
    STRIKECAST_ONREQ_TTL_SEC=120

WORKDIR /app

# OS deps for lightgbm + pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Source code
COPY src/ ./src/

# Slim runtime data — only what the API actually reads.
# The .dockerignore strips data/raw/ and the heavy unused parquets.
COPY data/processed/train.parquet                    ./data/processed/
COPY data/processed/games.parquet                    ./data/processed/
COPY data/processed/games_xref.parquet               ./data/processed/
COPY data/processed/odds_close.parquet               ./data/processed/
COPY data/processed/features_market.parquet          ./data/processed/
COPY data/processed/features_pitcher_season.parquet  ./data/processed/
COPY data/processed/daily_pick_signal_history.parquet ./data/processed/
COPY data/processed/features_circadian.parquet       ./data/processed/
COPY data/processed/features_travel.parquet          ./data/processed/
COPY data/processed/features_luck.parquet            ./data/processed/
COPY data/processed/features_cluster_luck.parquet    ./data/processed/
COPY data/processed/features_comeback.parquet        ./data/processed/
COPY data/processed/features_game_flow.parquet       ./data/processed/
COPY data/processed/features_burn.parquet            ./data/processed/
COPY data/processed/features_starter_workload.parquet ./data/processed/
COPY data/processed/features_high_leverage.parquet    ./data/processed/
COPY data/processed/features_catcher_control.parquet  ./data/processed/
COPY data/processed/value_picks.parquet               ./data/processed/

COPY data/models/ensemble.pkl     ./data/models/
COPY data/models/lgb_cls.pkl      ./data/models/
COPY data/models/lgb_reg.pkl      ./data/models/
COPY data/models/f5_cls.pkl       ./data/models/
COPY data/models/f5_reg.pkl       ./data/models/
COPY data/models/value_first.pkl  ./data/models/
COPY data/models/value_win.pkl    ./data/models/

EXPOSE 8000

# Railway/Fly inject $PORT — fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn src.serve.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
