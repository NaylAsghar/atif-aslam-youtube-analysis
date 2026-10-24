import os
 
import pandas as pd
 
from text_utils import THEMES, clean_text, detect_language, find_themes
 
RAW = "data/raw/comments_raw.csv"
OUT = "data/processed/comments_clean.csv"
 
 
def main():
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    start = len(df)
    print(f"Raw comments: {start}")
 
    df = df.drop_duplicates(subset="comment_id")
    df["text"] = df["text"].fillna("").astype(str)
    df["text"] = df["text"].str.replace(r"<[^>]+>", " ", regex=True)               # html tags
    df["text"] = df["text"].str.replace(r"https?://\S+|www\.\S+", " ", regex=True)  # links
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["song", "text"])   # copy-paste spam
 
    df["text_clean"] = df["text"].apply(clean_text)
    df["word_count"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)
    df["language"] = df["text"].apply(detect_language)
 
    themes = df["text_clean"].apply(find_themes)
    for theme in THEMES:
        df[f"theme_{theme}"] = themes.apply(lambda s, t=theme: t in s)
 
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["year"] = df["published_at"].dt.year
 
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Clean comments: {len(df)} (removed {start - len(df)})")
    print("\nComments per song:")
    print(df["song"].value_counts().to_string())
    print("\nLanguage mix:")
    print(df["language"].value_counts().to_string())
    print(f"\nSaved to {OUT}")
 
 
if __name__ == "__main__":
    main()