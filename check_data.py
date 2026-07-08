"""Quick verification of processed data after update_data.bat."""
import pandas as pd
from pathlib import Path
import datetime

processed = Path("data/processed")

g = pd.read_parquet(processed / "games.parquet")
g["game_date"] = pd.to_datetime(g["game_date"])
done = g[g["home_score"].notna()]

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
print("=" * 52)
if days_behind <= 1:
    print("  OK - Datos al dia!")
elif days_behind <= 3:
    print(f"  AVISO - Faltan {days_behind} dias de resultados.")
else:
    print(f"  ATRASADO - Faltan {days_behind} dias. Corre update_data.bat")
print()
