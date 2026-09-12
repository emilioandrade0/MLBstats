"""Quick verification of processed data after update_data.bat."""
import pandas as pd
from pathlib import Path
import datetime

processed = Path("data/processed")

g = pd.read_parquet(processed / "games.parquet")
g["game_date"] = pd.to_datetime(g["game_date"])
status = g["status"].astype(str).str.lower()
done = g[
    g["game_type"].eq("R")
    & status.str.match(r"^(final|game over|completed early)")
    & g["home_score"].notna()
    & g["away_score"].notna()
]

t = pd.read_parquet(processed / "train.parquet")
t["game_date"] = pd.to_datetime(t["game_date"])

fm = pd.read_parquet(processed / "features_market.parquet")

today = datetime.date.today()
last_done = done["game_date"].max().date()
days_behind = (today - last_done).days

print()
print("=" * 52)
print("  STRIKECAST | Estado de datos")
print("=" * 52)
print(f"  games.parquet   {len(g):>6,} juegos  (completados: {len(done):,})")
print(f"  Ultimo jugado:  {last_done}  ({days_behind}d atras)")
print(f"  train.parquet   {len(t):>6,} filas  ({t['game_date'].min().date()} -> {t['game_date'].max().date()})")
print(f"  features_market {len(fm):>6,} filas")
artifacts = [
    ("walkforward OOS", processed / "walkforward_preds.parquet"),
    ("componentes", Path("STARTFROMTHEEND/outputs/13_component_predictions.parquet")),
    ("simulador OOS", Path("STARTFROMTHEEND/outputs/14_linked_simulation_predictions.csv")),
]
artifact_lag = []
for label, path in artifacts:
    if not path.exists():
        print(f"  {label:<17} FALTA")
        artifact_lag.append(label)
        continue
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    column = "game_date" if "game_date" in frame.columns else "date"
    latest = pd.to_datetime(frame[column], errors="coerce").max().date()
    marker = "OK" if latest >= last_done else f"ATRASADO {last_done - latest}"
    print(f"  {label:<17} {latest}  {marker}")
    if latest < last_done:
        artifact_lag.append(label)
print("=" * 52)
if days_behind <= 1 and not artifact_lag:
    print("  OK - Datos al dia!")
elif artifact_lag:
    print("  ATRASADO - Falta actualizar: " + ", ".join(artifact_lag))
elif days_behind <= 3:
    print(f"  AVISO - Faltan {days_behind} dias de resultados.")
else:
    print(f"  ATRASADO - Faltan {days_behind} dias. Corre update_data.bat")
print()
