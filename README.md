# Sentiment and Audience Analysis of Atif Aslam's Music Using YouTube Comments

Big Data Analytics course project. Collects ~2,000 public YouTube comments from 5 Atif Aslam songs
using the YouTube Data API v3, then cleans and analyzes them.

## Setup
1. `python -m venv venv` and activate it
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your YouTube API key
4. Fill in the video IDs in `src/config.py`
5. `python src/scrape_comments.py`

## Structure
- `src/` scraping and analysis scripts
- `data/raw` and `data/processed` datasets
- `notebooks/` analysis notebook
- `reports/` project report

(Hypotheses, results and conclusions will be added as the project progresses.)
