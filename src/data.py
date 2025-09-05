import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from .config import DATA_DIR
from .utils import ensure_dirs

CSV_PATH = DATA_DIR / "ag_news.csv"

def _from_local_csv():
    df = pd.read_csv(CSV_PATH)
    assert {"title","description","label"} <= set(df.columns)
    return df

def _download_ag_news():
    ds = load_dataset("ag_news")
    # Combine title + description, keep label
    train = pd.DataFrame(ds["train"])
    test = pd.DataFrame(ds["test"])
    df = pd.concat([train, test], ignore_index=True)
    df.rename(columns={"text":"description"}, inplace=True)
    # dataset has 'text' (title+desc) and 'label'; we also split title heuristically when possible
    if "title" not in df.columns:
        # crude split: treat first sentence as title
        s = df["description"].str.split(". ", n=1, expand=True)
        df["title"] = s[0].fillna("")
        df["description"] = s[1].fillna(df["title"])
    df = df[["title","description","label"]]
    df.to_csv(CSV_PATH, index=False)
    return df

def load_data():
    ensure_dirs(DATA_DIR)
    if Path(CSV_PATH).exists():
        return _from_local_csv()
    try:
        return _download_ag_news()
    except Exception as e:
        # tiny fallback data
        data = {
            "title": [
                "Central bank raises rates",
                "Team wins championship",
                "Tech firm unveils new chip",
                "UN holds emergency meeting",
            ],
            "description": [
                "Markets react as central bank hikes interest rates amid inflation concerns.",
                "Fans celebrate after the underdogs clinch the national title in a thrilling final.",
                "The company introduced a faster processor aimed at AI workloads.",
                "Diplomats gather to discuss escalating tensions in the region.",
            ],
            "label": [2,1,3,0],  # Business, Sports, Sci/Tech, World
        }
        df = pd.DataFrame(data)
        df.to_csv(CSV_PATH, index=False)
        return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("Saved:", CSV_PATH)
