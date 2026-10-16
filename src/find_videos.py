"""Helper: search YouTube for a song and print candidate video IDs.
Usage: python src/find_videos.py "Atif Aslam Aadat official"
Pick the OFFICIAL upload (check channel name and view count).
"""
import os
import sys

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

query = " ".join(sys.argv[1:]) or "Atif Aslam Aadat"
response = youtube.search().list(
    q=query, part="snippet", type="video", maxResults=5
).execute()

for item in response["items"]:
    print(item["id"]["videoId"], "|", item["snippet"]["title"], "|", item["snippet"]["channelTitle"])
