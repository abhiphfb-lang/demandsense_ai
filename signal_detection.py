"""
01 - Signal Detection

Detect emerging market topics from external news (GDELT GKG) and,
best-effort, search interest (PyTrends).

Output: data/processed/topic_momentum.csv with a momentum score per topic,
consumed by gap_analysis.py and agent_orchestration.py.
"""

import os
import io
import zipfile
from datetime import datetime, timedelta
from collections import Counter

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

GKG_COLUMNS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "V1Counts", "V2Counts", "V1Themes", "V2Themes",
    "V1Locations", "V2Locations", "V1Persons", "V2Persons",
    "V1Organizations", "V2Organizations", "V1Tone", "GCAM",
    "SharingImage", "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds",
    "Quotations", "AllNames", "Amounts", "TranslationInfo", "Extras",
]

SEED_KEYWORDS = ["artificial intelligence", "supply chain", "generative ai"]


def get_latest_gkg_url():
    resp = requests.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt", timeout=30)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        if line.strip().endswith(".gkg.csv.zip"):
            return line.split(" ")[-1].strip()
    raise ValueError("No GKG file found in lastupdate.txt")


def download_gkg_at(url):
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(
                f, sep="\t", header=None, names=GKG_COLUMNS,
                dtype=str, on_bad_lines="skip",
            )
    return df


def fetch_recent_snapshots(n_files=8, step_minutes=15):
    """Walk back from the latest available GKG file, collecting up to n_files snapshots."""
    latest_url = get_latest_gkg_url()
    base = latest_url.rsplit("/", 1)[0]
    fname = latest_url.rsplit("/", 1)[1]
    ts_str, suffix = fname[:14], fname[14:]
    ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")

    snapshots = []
    attempts = 0
    while len(snapshots) < n_files and attempts < n_files * 3:
        candidate_url = f"{base}/{ts.strftime('%Y%m%d%H%M%S')}{suffix}"
        df = download_gkg_at(candidate_url)
        if df is not None:
            snapshots.append((ts, df))
            print("Fetched", candidate_url, "rows=", len(df))
        ts -= timedelta(minutes=step_minutes)
        attempts += 1
    return snapshots


def extract_themes(df):
    counter = Counter()
    for themes in df["V2Themes"].dropna():
        for item in themes.split(";"):
            theme = item.split(",")[0].strip()
            if theme:
                counter[theme] += 1
    return counter


def compute_gdelt_momentum(snapshots):
    snapshots_sorted = sorted(snapshots, key=lambda x: x[0])
    mid = len(snapshots_sorted) // 2
    earlier = snapshots_sorted[:mid] or snapshots_sorted
    later = snapshots_sorted[mid:]

    earlier_counts = Counter()
    for _, df in earlier:
        earlier_counts.update(extract_themes(df))

    later_counts = Counter()
    for _, df in later:
        later_counts.update(extract_themes(df))

    all_themes = set(earlier_counts) | set(later_counts)
    run_timestamp = snapshots_sorted[-1][0] if snapshots_sorted else datetime.utcnow()

    rows = []
    for theme in all_themes:
        e = earlier_counts.get(theme, 0)
        l = later_counts.get(theme, 0)
        momentum = (l - e) / (e + 1)
        rows.append({
            "run_timestamp": run_timestamp,
            "topic": theme,
            "earlier_count": e,
            "later_count": l,
            "momentum": momentum,
        })

    df = pd.DataFrame(rows).sort_values("momentum", ascending=False)
    df["source"] = "gdelt"
    return df, run_timestamp


def fetch_search_trends(run_timestamp, seed_keywords=SEED_KEYWORDS):
    """Best-effort only: Google may rate-limit unauthenticated pytrends calls."""
    trends_df = pd.DataFrame()
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(seed_keywords, timeframe="today 3-m")
        trends_df = pytrends.interest_over_time()
        trends_df["run_timestamp"] = run_timestamp
    except Exception as e:
        print("PyTrends unavailable (continuing without it):", e)
    return trends_df


def trends_to_momentum(trends_df, run_timestamp, seed_keywords=SEED_KEYWORDS):
    """Turn the search-interest time series into momentum rows on the same
    schema as the GDELT topics, so search demand and news demand are
    comparable signals."""
    if trends_df.empty:
        return pd.DataFrame()

    trend_rows = []
    for keyword in seed_keywords:
        if keyword not in trends_df.columns:
            continue
        series = trends_df[keyword].dropna()
        if series.empty:
            continue
        mid = len(series) // 2
        earlier = series.iloc[:mid] if mid else series
        later = series.iloc[mid:]
        e = float(earlier.mean())
        l = float(later.mean())
        trend_rows.append({
            "run_timestamp": run_timestamp,
            "topic": keyword,
            "earlier_count": e,
            "later_count": l,
            "momentum": (l - e) / (e + 1),
            "source": "search_trends",
        })
    return pd.DataFrame(trend_rows)


def append_csv(df, path, dedup_subset=None):
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, df], ignore_index=True)
        if dedup_subset:
            combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
    else:
        combined = df
    combined.to_csv(path, index=False)
    return combined


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    snapshots = fetch_recent_snapshots(n_files=8)
    topic_momentum_df, run_timestamp = compute_gdelt_momentum(snapshots)

    trends_df = fetch_search_trends(run_timestamp)
    trend_momentum_df = trends_to_momentum(trends_df, run_timestamp)
    if not trend_momentum_df.empty:
        topic_momentum_df = pd.concat([topic_momentum_df, trend_momentum_df], ignore_index=True)
        topic_momentum_df = topic_momentum_df.sort_values("momentum", ascending=False)

    momentum_path = os.path.join(PROCESSED_DIR, "topic_momentum.csv")
    combined_momentum = append_csv(
        topic_momentum_df, momentum_path, dedup_subset=["run_timestamp", "topic"]
    )
    print("Appended to:", momentum_path, "| total rows now:", len(combined_momentum))

    if not trends_df.empty:
        trends_path = os.path.join(PROCESSED_DIR, "search_trends.csv")
        combined_trends = append_csv(
            trends_df.reset_index(), trends_path, dedup_subset=["run_timestamp", "date"]
        )
        print("Appended to:", trends_path, "| total rows now:", len(combined_trends))

    return topic_momentum_df


if __name__ == "__main__":
    run()
