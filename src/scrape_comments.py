"""Scrape top-level comments for each song using the YouTube Data API v3.

Run from the project root:  python src/scrape_comments.py
Output: data/raw/comments_raw.csv and data/raw/videos_metadata.csv
"""
import hashlib
import os

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import COMMENTS_PER_SONG, VIDEOS

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise SystemExit("YOUTUBE_API_KEY not found. Create a .env file (see .env.example).")

youtube = build("youtube", "v3", developerKey=API_KEY)


def anonymize(name: str) -> str:
    """Hash the username so we do not store personal identities (ethics)."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]


def get_video_metadata(song: str, video_id: str) -> dict:
    resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
    item = resp["items"][0]
    stats = item["statistics"]
    return {
        "song": song,
        "video_id": video_id,
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "published_at": item["snippet"]["publishedAt"],
        "view_count": int(stats.get("viewCount", 0)),
        "like_count": int(stats.get("likeCount", 0)),
        "total_comment_count": int(stats.get("commentCount", 0)),
    }


def get_comments(song: str, video_id: str, limit: int) -> list:
    comments, page_token = [], None
    while len(comments) < limit:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,          # API maximum per request
            order="relevance",       # use "time" for newest first
            textFormat="plainText",
            pageToken=page_token,
        ).execute()

        for item in resp["items"]:
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "song": song,
                "video_id": video_id,
                "comment_id": item["id"],
                "author_hash": anonymize(top["authorDisplayName"]),
                "text": top["textDisplay"],
                "like_count": top["likeCount"],
                "reply_count": item["snippet"]["totalReplyCount"],
                "published_at": top["publishedAt"],
            })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return comments[:limit]


def main():
    all_comments, all_meta = [], []
    for song, video_id in VIDEOS.items():
        if video_id.startswith("PASTE"):
            print(f"Skipping {song}: video ID not set in src/config.py")
            continue
        try:
            all_meta.append(get_video_metadata(song, video_id))
            rows = get_comments(song, video_id, COMMENTS_PER_SONG)
            all_comments.extend(rows)
            print(f"{song}: collected {len(rows)} comments")
        except HttpError as e:
            print(f"{song}: API error -> {e}")

    pd.DataFrame(all_comments).to_csv("data/raw/comments_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_meta).to_csv("data/raw/videos_metadata.csv", index=False, encoding="utf-8-sig")
    print(f"Done. Total comments: {len(all_comments)}")


if __name__ == "__main__":
    main()
