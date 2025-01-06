# YouTube Summaries & Cross-Reference API

A FastAPI application that:
1. Fetches transcripts of YouTube videos.
2. Summarizes each video’s content (optionally filtered by a given topic).
3. (Optionally) Cross-references multiple summaries to produce a combined summary.
4. Uses **Anthropic’s Claude** LLM for text analysis and summarization.

---

## Features

- **YouTube Transcripts**: Automatically fetches transcripts using [`youtube_transcript_api`](https://pypi.org/project/youtube-transcript-api/).
- **Summarization**: Generates summary text using Anthropic’s Claude API.
- **Cross-Referencing**: Uses [Chroma](https://github.com/chroma-core/chroma) with sentence embeddings to find the most thematically similar content, then merges those summaries with Claude again.
- **Topic Filtering**: Optionally filters transcripts by a specific topic keyword.

---

## Installation

**Clone** this repository:
   ```bash
   git clone https://github.com/<your_username>/<repo_name>.git
   cd <repo_name>
   ```

---

## Install dependencies:

   ```bash
  pip install -r requirements.txt
  ```
  
Ensure you have a valid Python 3.9+ environment.

---

## Running the Server
Use Uvicorn to run the FastAPI application:

```bash
uvicorn main:app --reload
```
By default, the server runs on http://127.0.0.1:8000 on localhost. You can access a FASTAPI playground by going to http://127.0.0.1:8000/docs

---
## Usage
Endpoint: POST /process_videos\
Example cURL Call - Request Body (JSON):

```json
{
  "youtube_urls": [
    "https://www.youtube.com/watch?v=btlmHbwwqLg",
    "https://www.youtube.com/watch?v=RoO9nArroVc"
  ],
  "claude_api_key": "<INSERT_KEY>",
  "topic": "cars",
  "cross_reference": true
}
```
youtube_urls: A list of YouTube links to summarize.\
claude_api_key: Your Anthropic Claude API key.\
topic: (Optional) A keyword that helps filter the transcript to relevant sentences.\
cross_reference: A boolean indicating whether to generate a combined summary across videos.\



```bash
curl -X POST http://127.0.0.1:8000/process_videos \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_urls": [
      "https://www.youtube.com/watch?v=btlmHbwwqLg",
      "https://www.youtube.com/watch?v=RoO9nArroVc"
    ],
    "claude_api_key": "<INSERT_KEY>",
    "topic": "cars",
    "cross_reference": true
  }'
```

Example Response
```json
{
  "summaries": [
    "The text is a review and discussion about the 2024 BMW X5 M60i...",
    "The text discusses the Mercedes-Benz GLE luxury SUV..."
  ],
  "combined_summary": "Here is a combined summary of the two video reviews..."
}
summaries: Individual summaries for each YouTube video.
combined_summary: If cross_reference=true and multiple videos were processed, a merged summary comparing or contrasting the videos.
```
