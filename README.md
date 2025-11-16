Video Relevance Evaluator (Streamlit prototype)
A simple Streamlit app that takes a YouTube URL or an uploaded .txt transcript, compares the content with a user‑provided title & description, and returns:
an overall relevance score (0‑100 %)
per‑segment relevance scores with a colour‑coded heat‑map
a flag for promotional language
a short LLM‑generated explanation

1. Prerequisites
Python 3.9‑3.12
2. Install dependencies
pip install -r requirements.txt
requirements.txt
Code
streamlit
pandas
torch
transformers
sentence-transformers
youtube-transcript-api
plotly
google-cloud-aiplatform  # for Gemini 1.5‑Flash
3. Run locally

streamlit run uiapp.py
The app will open on http://localhost:8501.
4. Usage
YouTube URL – paste a valid YouTube link (e.g., https://youtu.be/pOxlJ0RunA8).
Transcript file – upload a plain‑text file (.txt).
Video Title – enter the expected title.
Description – brief description of the video.
Chunk size – number of words per segment (default 50).
Off‑topic threshold – % below which a segment is flagged (default 30).
Click “Analyse” to see the relevance score, heat‑map, and explanation.
5. Demo
A short walkthrough video is available here:
6. Deployment (optional)
If you want to host the app:
Streamlit Community Cloud – push the repo, add secrets (YOUTUBE_API_KEY, GEMINI_API_KEY), and deploy.
Docker – build with the provided Dockerfile and run on any container platform.
Dockerfile
Docker
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY uiapp.py .
EXPOSE 8501
CMD ["streamlit", "run", "uiapp.py"]
7. Environment variables
Variable	Description
YOUTUBE_API_KEY	YouTube Data API key
GEMINI_API_KEY	API key for Google Gemini 1.5‑Flash
OFF_TOPIC_THRESHOLD	Optional override (e.g., “25”)
Add them to your deployment’s config or a .env file (do not commit real keys).
