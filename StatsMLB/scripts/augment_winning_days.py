from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


APP = Path(__file__).resolve().parents[1]
PUBLIC_DATA = APP / "public" / "data"


def main():
    manifest_path = PUBLIC_DATA / "walkforward.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    combinations = {int(item["mask"]): item for item in manifest["combinations"]}
    masks = np.array(sorted(combinations), dtype=int)
    winning = np.zeros(len(masks), dtype=int)
    even = np.zeros(len(masks), dtype=int)
    losing = np.zeros(len(masks), dtype=int)
    perfect = np.zeros(len(masks), dtype=int)
    fifteen_winning = np.zeros(len(masks), dtype=int)
    fifteen_correct = np.zeros(len(masks), dtype=int)
    fifteen_perfect = np.zeros(len(masks), dtype=int)
    days = 0
    fifteen_days = 0

    for filename in manifest.get("dayFiles", []):
        payload = json.loads((PUBLIC_DATA / filename).read_text(encoding="utf-8"))
        for day in payload.get("days", {}).values():
            games = day.get("games", [])
            if not games:
                continue
            packed = np.vstack([
                np.frombuffer(base64.b64decode(game["maskProbabilitiesEncoded"]), dtype=np.uint8)
                for game in games
            ])
            actual_home = np.array([game["actualWinner"] == game["home"] for game in games], dtype=bool)
            correct = ((packed[:, masks] >= 128) == actual_home[:, None]).sum(axis=0)
            total = len(games)
            winning += correct > total / 2
            even += correct == total / 2
            losing += correct < total / 2
            perfect += correct == total
            days += 1
            if total == 15:
                fifteen_winning += correct >= 8
                fifteen_correct += correct
                fifteen_perfect += correct == 15
                fifteen_days += 1

    for index, mask in enumerate(masks):
        combinations[int(mask)].update({
            "days": int(days),
            "winningDays": int(winning[index]),
            "evenDays": int(even[index]),
            "losingDays": int(losing[index]),
            "perfectDays": int(perfect[index]),
            "fifteenGameDays": int(fifteen_days),
            "fifteenGameWinningDays": int(fifteen_winning[index]),
            "fifteenGameGames": int(fifteen_days * 15),
            "fifteenGameCorrect": int(fifteen_correct[index]),
            "fifteenGameAccuracy": float(fifteen_correct[index] / (fifteen_days * 15)) if fifteen_days else 0.0,
            "fifteenGamePerfectDays": int(fifteen_perfect[index]),
        })
    best = min(
        combinations.values(),
        key=lambda item: (-item["winningDays"], -item["accuracy"], bin(int(item["mask"])).count("1"), item["mask"]),
    )
    manifest["bestWinningDaysMask"] = int(best["mask"])
    best_fifteen = min(
        combinations.values(),
        key=lambda item: (
            -item["fifteenGameAccuracy"],
            -item["fifteenGameWinningDays"],
            -item["accuracy"],
            bin(int(item["mask"])).count("1"),
            item["mask"],
        ),
    )
    manifest.pop("bestFifteenGameDaysMask", None)
    manifest["bestFifteenGameAccuracyMask"] = int(best_fifteen["mask"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "mask": best["mask"],
        "winningDays": best["winningDays"],
        "days": best["days"],
        "accuracy": best["accuracy"],
        "correct": best["correct"],
        "games": best["games"],
        "bestFifteenGameAccuracy": {
            "mask": best_fifteen["mask"],
            "winningDays": best_fifteen["fifteenGameWinningDays"],
            "days": best_fifteen["fifteenGameDays"],
            "accuracy": best_fifteen["fifteenGameAccuracy"],
            "correct": best_fifteen["fifteenGameCorrect"],
            "games": best_fifteen["fifteenGameGames"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
