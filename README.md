# Conversational Meeting Assistant

Bot joins a Google Meet, records the audio (Windows WASAPI loopback), transcribes it
(Whisper via Groq, or Sarvam), ingests the transcript into a per-meeting ChromaDB store,
and lets you chat with the meeting through a RAG pipeline (MiniLM embeddings + Groq LLM) in Streamlit.

## Setup (Windows)
    python -m venv venv && venv\Scripts\activate
    pip install -r requirements.txt
    playwright install chromium
    copy .env.example .env      # add GROQ_API_KEY (and SARVAM_API_KEY if used)

## Run
    python meeting_bot.py https://meet.google.com/xxx-xxxx-xxx --transcriber whisper
    # let the bot be admitted, press Enter when the meeting ends
    streamlit run app2.py

## Pipeline
Google Meet (Playwright) -> loopback audio -> STT -> JSON -> chunk (1000/200) ->
embeddings -> ChromaDB (cosine) -> retrieve -> filter/dedupe -> LLM answer + sources
