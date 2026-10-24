import json
import os
 
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
 
from text_utils import SONG_RELEASE_YEAR, THEMES
 
IN = "data/processed/comments_scored.csv"
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)
 
df = pd.read_csv(IN, encoding="utf-8-sig")
df["engagement"] = df["like_count"] + df["reply_count"]
ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2e9e5b", "neutral": "#9aa0a6", "negative": "#d64545"}
results = {"total_comments": int(len(df))}
 
 
def save(name):
    plt.tight_layout()
    plt.savefig(f"images/{name}.png", dpi=150)
    plt.close()
 
 
# ---- Chart 1: sentiment by song ------------------------------------------------
share = pd.crosstab(df["song"], df["sentiment"], normalize="index").reindex(columns=ORDER).fillna(0) * 100
share.plot(kind="bar", stacked=True, color=[COLORS[c] for c in ORDER], figsize=(8, 5))
plt.title("Sentiment of comments by song"); plt.ylabel("% of comments"); plt.xlabel("")
plt.xticks(rotation=20, ha="right"); plt.legend(title="Sentiment")
save("1_sentiment_by_song")
 
# ---- Chart 2: engagement by sentiment -----------------------------------------
by_sent = df.groupby("sentiment")["engagement"].agg(["count", "mean", "median"]).reindex(ORDER)
by_sent[["mean", "median"]].plot(kind="bar", figsize=(7, 5), color=["#3b6fd4", "#8fb0ee"])
plt.title("Engagement (likes + replies) per comment, by sentiment")
plt.ylabel("Average / median engagement"); plt.xlabel(""); plt.xticks(rotation=0)
save("2_engagement_by_sentiment")
results["engagement_by_sentiment"] = by_sent.round(2).fillna(0).to_dict(orient="index")
 
# ---- Chart 3: language share ---------------------------------------------------
lang = df["language"].value_counts(normalize=True) * 100
lang.plot(kind="bar", color="#7a5cd6", figsize=(7, 5))
plt.title("Language of comments"); plt.ylabel("% of comments"); plt.xlabel(""); plt.xticks(rotation=0)
save("3_language_share")
results["language_share_pct"] = lang.round(1).to_dict()
 
# ---- Chart 4: engagement by theme ----------------------------------------------
theme_rows = []
for theme in THEMES:
    sub = df[df[f"theme_{theme}"]]
    theme_rows.append({"theme": theme, "comments": int(len(sub)),
                       "share_pct": round(len(sub) / len(df) * 100, 1),
                       "mean_engagement": round(float(sub["engagement"].mean()), 2) if len(sub) else 0.0,
                       "median_engagement": float(sub["engagement"].median()) if len(sub) else 0.0})
themes_df = pd.DataFrame(theme_rows).set_index("theme")
themes_df["mean_engagement"].plot(kind="bar", color="#e08a2e", figsize=(8, 5))
plt.axhline(df["engagement"].mean(), color="black", ls="--", label="All comments (average)")
plt.title("Average engagement of comments, by theme mentioned")
plt.ylabel("Average engagement"); plt.xlabel(""); plt.xticks(rotation=20, ha="right"); plt.legend()
save("4_engagement_by_theme")
results["themes"] = themes_df.to_dict(orient="index")
 
# ---- Chart 5: comment length vs likes ------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(df["word_count"], np.log1p(df["like_count"]), alpha=0.35, s=14, color="#3b6fd4")
plt.title("Comment length vs likes"); plt.xlabel("Words in comment"); plt.ylabel("log(1 + likes)")
save("5_length_vs_likes")
 
# ---- Per-song engagement table ---------------------------------------------------
by_song = df.groupby("song").agg(comments=("engagement", "size"),
                                 mean_engagement=("engagement", "mean"),
                                 median_engagement=("engagement", "median"),
                                 pct_positive=("sentiment", lambda s: (s == "positive").mean() * 100)).round(2)
results["by_song"] = by_song.to_dict(orient="index")
 
# ---- Hypothesis tests --------------------------------------------------------------
hyp = {}
 
# H1: positive comments are the largest group
counts = df["sentiment"].value_counts()
pos_share = counts.get("positive", 0) / len(df) * 100
largest = counts.idxmax()
p1 = stats.binomtest(int(counts.get("positive", 0)), len(df), 0.5, alternative="greater").pvalue
hyp["H1"] = {"claim": "Positive comments are the largest sentiment group",
             "positive_pct": round(pos_share, 1), "largest_group": largest,
             "p_value_positive_is_majority": float(p1),
             "verdict": "SUPPORTED" if largest == "positive" else "NOT SUPPORTED"}
 
# H2: older songs get more nostalgic comments than newer songs
df["release_year"] = df["song"].map(SONG_RELEASE_YEAR)
known = df.dropna(subset=["release_year"]).copy()
known["era"] = np.where(known["release_year"] <= 2010, "older", "newer")
if known["era"].nunique() == 2:
    table = pd.crosstab(known["era"], known["theme_nostalgia"])
    chi2, p2, _, _ = stats.chi2_contingency(table)
    rates = known.groupby("era")["theme_nostalgia"].mean() * 100
    hyp["H2"] = {"claim": "Older songs (<=2010) get more nostalgic comments than newer songs",
                 "nostalgic_pct_older": round(float(rates["older"]), 1),
                 "nostalgic_pct_newer": round(float(rates["newer"]), 1),
                 "p_value": float(p2),
                 "verdict": "SUPPORTED" if rates["older"] > rates["newer"] and p2 < 0.05 else "NOT SUPPORTED"}
else:
    hyp["H2"] = {"claim": "Older vs newer nostalgia", "verdict": "NOT TESTABLE (need both older and newer songs)"}
 
# H3: Roman Urdu is the most common language
top_lang = lang.idxmax()
hyp["H3"] = {"claim": "Roman Urdu is the most common comment language",
             "top_language": top_lang, "top_language_pct": round(float(lang.max()), 1),
             "verdict": "SUPPORTED" if top_lang == "Roman Urdu" else "NOT SUPPORTED"}
 
# H4: engagement differs by sentiment, and length relates to likes
groups = [df.loc[df["sentiment"] == s, "engagement"] for s in ORDER if (df["sentiment"] == s).sum() > 1]
kw_p = float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else float("nan")
rho, p_rho = stats.spearmanr(df["word_count"], df["like_count"])
hyp["H4"] = {"claim": "Sentiment and comment length relate to engagement",
             "kruskal_p_value_sentiment": kw_p,
             "spearman_rho_length_vs_likes": round(float(rho), 3), "spearman_p_value": float(p_rho),
             "verdict": "SUPPORTED" if (kw_p < 0.05 or p_rho < 0.05) else "NOT SUPPORTED"}
results["hypotheses"] = hyp
 
with open("reports/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=float)
 
# ---- Human-readable summary --------------------------------------------------------
lines = ["# Results summary", "", f"Total comments analysed: **{len(df)}**", "",
         "## Engagement by sentiment", "```\n" + by_sent.round(2).to_string() + "\n```", "",
         "## Engagement by song", "```\n" + by_song.to_string() + "\n```", "",
         "## Themes", "```\n" + themes_df.to_string() + "\n```", "", "## Hypotheses"]
for key, h in hyp.items():
    lines.append(f"- **{key}** {h['claim']}: **{h['verdict']}**")
    for k, v in h.items():
        if k not in ("claim", "verdict"):
            lines.append(f"    - {k}: {v}")
with open("reports/results_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
 
print("\n".join(lines))
print("\nCharts saved in images/, results in reports/")