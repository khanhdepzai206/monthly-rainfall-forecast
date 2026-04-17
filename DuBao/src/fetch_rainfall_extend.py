# -*- coding: utf-8 -*-
"""
Extend rainfall daily dataset (raw_daily.csv) to 2025 using Open-Meteo Archive API.

Input  : ../data/raw_daily.csv (2 header rows, then date,rainfall)
Output : overwrite ../data/raw_daily.csv (keeps same 2 header rows)
Backup : ../data/raw_daily_legacy.csv (created once)

We fetch Open-Meteo daily precipitation_sum for missing dates.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Iterable, List, Tuple

import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _read_raw_daily_csv(path: str) -> pd.DataFrame:
    """Read raw_daily.csv with 2 header rows."""
    df = pd.read_csv(path, skiprows=2, header=None, names=["date", "rainfall"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
    df = df.dropna(subset=["date", "rainfall"]).sort_values("date").reset_index(drop=True)
    return df


def _write_raw_daily_csv_with_headers(path: str, df: pd.DataFrame) -> None:
    """Write with the exact 2 header rows expected by preprocess.load_raw_daily."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d 00:00:00")
    df = df[["date", "rainfall"]]

    header_lines = ["Mưa(mm),Unnamed: 1", "Thờigian,Pr_DaNang"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
    df.to_csv(path, mode="a", index=False, header=False, encoding="utf-8")


def _daterange_chunks(start: date, end: date, chunk_days: int = 366) -> Iterable[Tuple[date, date]]:
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + timedelta(days=chunk_days - 1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_precipitation_sum_daily(
    start_date: str,
    end_date: str,
    lat: float = 16.0678,
    lon: float = 108.2208,
) -> pd.DataFrame:
    """Fetch daily precipitation_sum for date range inclusive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["precipitation_sum"],
        "timezone": "Asia/Ho_Chi_Minh",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    d = data.get("daily") or {}
    times = d.get("time") or []
    vals = d.get("precipitation_sum") or []
    out = pd.DataFrame({"date": times, "rainfall": vals})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["rainfall"] = pd.to_numeric(out["rainfall"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def extend_raw_daily_to_year(
    raw_daily_path: str,
    target_end: date = date(2025, 12, 31),
    lat: float = 16.0678,
    lon: float = 108.2208,
) -> None:
    if not os.path.exists(raw_daily_path):
        raise FileNotFoundError(f"Missing: {raw_daily_path}")

    df = _read_raw_daily_csv(raw_daily_path)
    if df.empty:
        raise ValueError("raw_daily.csv is empty after parsing.")

    last_date = pd.Timestamp(df["date"].max()).date()
    if last_date >= target_end:
        print(f"No extend needed. raw_daily last_date={last_date} >= {target_end}")
        return

    fetch_start = last_date + timedelta(days=1)
    fetch_end = target_end
    print(f"Extending rainfall from {fetch_start} to {fetch_end} ...")

    parts: List[pd.DataFrame] = []
    for s, e in _daterange_chunks(fetch_start, fetch_end, chunk_days=366):
        part = fetch_precipitation_sum_daily(
            s.strftime("%Y-%m-%d"),
            e.strftime("%Y-%m-%d"),
            lat=lat,
            lon=lon,
        )
        parts.append(part)
        print(f"  fetched {s}..{e}: {len(part)} rows")

    ext = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "rainfall"])
    ext = ext.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")

    merged = pd.concat([df, ext], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Backup once
    legacy = os.path.join(os.path.dirname(raw_daily_path), "raw_daily_legacy.csv")
    if not os.path.exists(legacy):
        import shutil

        shutil.copy2(raw_daily_path, legacy)
        print(f"Backup created: {legacy}")

    _write_raw_daily_csv_with_headers(raw_daily_path, merged)
    print(f"Updated raw_daily.csv: {raw_daily_path} (rows={len(merged)}, last_date={merged['date'].max().date()})")


def main():
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw_daily.csv")
    extend_raw_daily_to_year(os.path.abspath(raw_path))


if __name__ == "__main__":
    main()

