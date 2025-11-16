import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer, util

import google.generativeai as genai

# ---------------------------------------------------------
# Load environment variables safely
# ---------------------------------------------------------
load_dotenv()  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please set it in your .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------
# Utility: Extract YouTube transcript
# ---------------------------------------------------------
def get_transcript_from_youtube(url):
    try:
        video_id = url.split("v=")[-1] if "v=" in url else url.split("/")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t["text"] for t in transcript])
        return text
    except Exception as e:
        return None

# ---------------------------------------------------------
# Utility: Chunk text into pieces
# ---------------------------------------------------------
def chunk_text(text, chunk_size=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ---------------------------------------------------------
# Utility: Detect promotional language
# ---------------------------------------------------------
def contains_promotional_language(text):
    promo_keywords = [
        "subscribe", "like the video", "buy now", "limited time", 
        "special offer", "promotion", "discount", "click the link"
    ]
    text_lower = text.lower()
    return any(k in text_lower for k in promo_keywords)

# ---------------------------------------------------------
# Utility: Generate LLM explanation
# ---------------------------------------------------------
def generate_explanation(title, description, transcript):
    try:
        prompt = f"""
        You are an evaluator. Analyse how well the transcript matches the title and description.

        Title: {title}
        Description: {description}
        Transcript: {transcript[:2500]}

        Provide a short explanation (120 words max).
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return response.text
    except:
        return "Explanation unavailable (LLM error)."

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.set_page_config(page_title="Video Relevance Evaluator")

st.title("🎯 Video Relevance Evaluator")
st.write("Compare a YouTube video (or transcript) with your expected title & description.")

# Inputs
youtube_url = st.text_input("🔗 YouTube URL (optional)")
uploaded_file = st.file_uploader("📄 Upload Transcript (.txt)", type=["txt"])

title = st.text_input("📝 Expected Title")
description = st.text_area("📘 Expected Description")

chunk_size = st.slider("Chunk Size (words)", 20, 150, 50)
threshold = st.slider("Off-topic Threshold (%)", 1, 100, 30)

# ---------------------------------------------------------
# PROCESS BUTTON
# ---------------------------------------------------------
if st.button("🚀 Analyse"):
    # Step 1: Load text
    if youtube_url:
        transcript = get_transcript_from_youtube(youtube_url)
        if not transcript:
            st.error("Could not retrieve transcript from YouTube.")
            st.stop()
    elif uploaded_file:
        transcript = uploaded_file.read().decode("utf-8")
    else:
        st.error("Please provide a YouTube URL or upload a transcript.")
        st.stop()

    # Step 2: Chunk transcript
    chunks = chunk_text(transcript, chunk_size)

    # Step 3: Embeddings
    ref_text = f"{title}. {description}"
    ref_embedding = embed_model.encode(ref_text)

    results = []
    for idx, ch in enumerate(chunks):
        ch_emb = embed_model.encode(ch)
        sim = util.cos_sim(ref_embedding, ch_emb).item()
        sim_percent = round(sim * 100, 2)

        results.append({
            "Segment": idx + 1,
            "Text": ch,
            "Relevance (%)": sim_percent
        })

    df = pd.DataFrame(results)

    # Step 4: Compute overall score
    overall_score = df["Relevance (%)"].mean()

    # Step 5: Detect promotional language
    is_promotional = contains_promotional_language(transcript)

    # Step 6: Show results
    st.subheader("📊 Overall Relevance Score")
    st.metric(label="Score (0–100%)", value=f"{overall_score:.2f}%")

    if is_promotional:
        st.warning("⚠ Promotional language detected")

    st.subheader("🟦 Segment-level Heatmap")
    fig = px.bar(
        df,
        x="Segment",
        y="Relevance (%)",
        color="Relevance (%)",
        title="Relevance Breakdown by Segment",
        range_y=[0, 100]
    )
    st.plotly_chart(fig)

    st.subheader("📄 Detailed Segments")
    st.dataframe(df)

    st.subheader("🤖 LLM Explanation")
    with st.spinner("Generating explanation..."):
        explanation = generate_explanation(title, description, transcript)
    st.write(explanation)
