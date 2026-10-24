import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
from text_utils import NEUTRALIZED_WORDS, SENTIMENT_EXTRAS
 
IN = "data/processed/comments_clean.csv"
OUT = "data/processed/comments_scored.csv"
 
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(SENTIMENT_EXTRAS)
for word in NEUTRALIZED_WORDS:
    analyzer.lexicon[word] = 0.0
 
 
def label(compound):
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"
 
 
def main():
    df = pd.read_csv(IN, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    df["compound"] = df["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["sentiment"] = df["compound"].apply(label)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
 
    print("Overall sentiment:")
    print((df["sentiment"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nSentiment per song (%):")
    print((pd.crosstab(df["song"], df["sentiment"], normalize="index") * 100).round(1).to_string())
    print(f"\nSaved to {OUT}")
 
 
if __name__ == "__main__":
    main()