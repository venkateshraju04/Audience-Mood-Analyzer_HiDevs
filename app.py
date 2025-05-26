import praw, re, json, emoji, cohere
import streamlit as st
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import nltk
from dotenv import load_dotenv
import os

# Download NLTK data if not already present
nltk.download("punkt")
nltk.download("stopwords")

# Load environment variables
load_dotenv()

# Reddit API keys from .env
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_API_KEY"),
    client_secret=os.getenv("REDDIT_SECRET_KEY"),
    user_agent=os.getenv("USER_AGENT")
)

# Cohere API from .env
co = cohere.Client("HIPZcL01IEuowrHCCluAD6VEEWmlDmmDcpFCiARj")

def extract_comments_from_post(post_url):
    submission = reddit.submission(url=post_url)
    submission.comments.replace_more(limit=0)
    comments_data = []
    for comment in submission.comments.list():
        comments_data.append({
            "comment_id": comment.id,
            "author": str(comment.author),
            "body": comment.body,
            "score": comment.score,
            "created_utc": comment.created_utc
        })
    return comments_data

def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text.lower())
    return " ".join([t for t in tokens if t not in stopwords.words("english")])

# Sentiment model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

label_map = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

def add_sentiment_to_comments(comments):
    for c in comments:
        try:
            c["cleaned"] = clean_text(c["body"])
            result = sentiment_pipeline(c["cleaned"][:512])[0]
            c["sentiment"] = label_map.get(result["label"], "UNKNOWN")
            c["sentiment_score"] = round(result["score"], 4)
        except Exception as e:
            c["sentiment"] = "ERROR"
            c["sentiment_score"] = 0.0
    return comments

def calculate_sentiment_percentages(comments):
    sentiment_counts = Counter([c["sentiment"] for c in comments])
    total = sum(sentiment_counts.values())
    return {k: round((v / total) * 100, 2) for k, v in sentiment_counts.items()}

def summarize_comments_with_cohere(comments, sentiment_type="POSITIVE"):
    text_block = " ".join([c["cleaned"] for c in comments if c["sentiment"] == sentiment_type])
    if len(text_block.strip()) < 250:
        return f"Not enough {sentiment_type.lower()} comments to summarize properly."

    response = co.summarize(
        text=text_block[:4000],
        model="summarize-xlarge",
        length="medium",
        format="paragraph",
        temperature=0.3,
        extractiveness="medium"
    )
    return response.summary


# Streamlit UI
st.title("📊 Reddit Post Sentiment Analyzer")

post_url = st.text_input("Enter Reddit Post URL:")
if st.button("Analyze") and post_url:
    with st.spinner("Extracting and analyzing comments..."):
        comments = extract_comments_from_post(post_url)
        comments = add_sentiment_to_comments(comments)

        with open("reddit_sentiment_output.json", "w", encoding="utf-8") as f:
            json.dump(comments, f, indent=2)

        percents = calculate_sentiment_percentages(comments)
        pos_summary = summarize_comments_with_cohere(comments, "POSITIVE")
        neg_summary = summarize_comments_with_cohere(comments, "NEGATIVE")

    st.subheader("Sentiment Breakdown:")
    st.json(percents)

    st.subheader("🟢 Positive Summary:")
    st.write(pos_summary)

    st.subheader("🔴 Negative Summary:")
    st.write(neg_summary)
