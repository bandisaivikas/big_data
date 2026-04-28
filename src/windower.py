import pandas as pd
import os

def create_windows(parquet_path: str, window_size_days: int = 30, step_size_days: int = 15):
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    start = df["date"].min()
    end = df["date"].max()
    
    windows = []
    current = start
    
    while current + pd.Timedelta(days=window_size_days) <= end:
        window_end = current + pd.Timedelta(days=window_size_days)
        window_df = df[(df["date"] >= current) & (df["date"] < window_end)]
        
        if len(window_df) >= 10:
            windows.append({
                "start": current,
                "end": window_end,
                "texts": window_df["text"].tolist()
            })
        current += pd.Timedelta(days=step_size_days)
    
    print(f"Created {len(windows)} windows of {window_size_days} days each")
    for i, w in enumerate(windows[:3]):
        print(f"  Window {i+1}: {w['start'].date()} to {w['end'].date()} — {len(w['texts'])} articles")
    print("  ...")
    return windows

if __name__ == "__main__":
    windows = create_windows("data/raw/corpus.parquet")
