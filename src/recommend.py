import os
 
import pandas as pd
 
from text_utils import THEMES
 
IN = "data/processed/comments_scored.csv"
MIN_GROUP = 30   # ignore groups with fewer comments than this
 
df = pd.read_csv(IN, encoding="utf-8-sig")
df["engagement"] = df["like_count"] + df["reply_count"]
overall = df["engagement"].mean()
 
THEME_ADVICE = {
    "nostalgia": "throwback / 'remember when' content (old songs, memories, then-and-now posts)",
    "love_romance": "romantic and love-themed content (couples, dedications, Valentine-style posts)",
    "heartbreak_sad": "emotional, heartbreak-themed content (slow, sad, relatable stories)",
    "voice_praise": "content that spotlights the voice and talent (live clips, behind-the-scenes, acoustic cuts)",
    "pakistan_india": "cross-border, 'love from Pakistan and India' content that unites both audiences",
}
 
 
def weak(mean):
    """Warn when a group is barely different from the overall average."""
    if overall and abs(mean - overall) / overall < 0.10:
        return " (Note: only a small difference from the overall average, so this is weak evidence.)"
    return ""
 
 
def best_group(series_groups):
    valid = {k: v for k, v in series_groups if len(v) >= MIN_GROUP}
    if not valid:
        return None
    key = max(valid, key=lambda k: valid[k].mean())
    return key, valid[key]
 
 
recs = []
 
# 1. Which song / content?
song_pick = best_group((s, g["engagement"]) for s, g in df.groupby("song"))
if song_pick:
    song, vals = song_pick
    recs.append(f"**Re-promote '{song}'.** Its comments earn the most engagement "
                f"(average {vals.mean():.1f} likes+replies per comment vs {overall:.1f} overall). "
                f"Clips, remixes or anniversary posts of this song should perform well." + weak(vals.mean()))
 
# 2. Which theme?
theme_groups = [(t, df.loc[df[f"theme_{t}"], "engagement"]) for t in THEMES]
theme_pick = best_group(theme_groups)
if theme_pick:
    theme, vals = theme_pick
    recs.append(f"**Post {THEME_ADVICE[theme]}.** Comments about this theme average "
                f"{vals.mean():.1f} engagement ({len(vals)} comments, {len(vals)/len(df)*100:.0f}% of the audience talks about it)." + weak(vals.mean()))
 
# 3. Which tone? (sentiment)
sent_pick = best_group((s, g["engagement"]) for s, g in df.groupby("sentiment"))
if sent_pick:
    sent, vals = sent_pick
    recs.append(f"**Aim for a '{sent}' tone.** {sent.capitalize()} comments get the highest average engagement "
                f"({vals.mean():.1f}), so content that triggers that reaction spreads further." + weak(vals.mean()))
 
# 4. Which language?
lang_pick = best_group((l, g["engagement"]) for l, g in df.groupby("language"))
top_lang = df["language"].value_counts().idxmax()
if lang_pick:
    lang, vals = lang_pick
    recs.append(f"**Write captions in {top_lang}** (the most common comment language, "
                f"{(df['language'] == top_lang).mean() * 100:.0f}% of comments). "
                f"Comments in {lang} get the highest average engagement ({vals.mean():.1f})." + weak(vals.mean()))
 
# 5. Always: community management
recs.append("**Reply to the most-liked comments.** Highly liked comments show what the audience cares about; "
            "answering them early keeps the conversation (and engagement) growing.")
 
text = ["# Recommendation engine: what to post next", "",
        f"Based on {len(df)} public YouTube comments (engagement = likes + replies).", ""]
text += [f"{i}. {r}" for i, r in enumerate(recs, 1)]
text += ["", "_Limitations: small sample, engagement is skewed by a few viral comments, and "
         "sentiment tools are less accurate on Roman Urdu._"]
 
os.makedirs("reports", exist_ok=True)
with open("reports/recommendations.md", "w", encoding="utf-8") as f:
    f.write("\n".join(text))
print("\n".join(text))
 