from typing import List, Optional
import chromadb
from anthropic import Anthropic
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(youtube_url: str) -> str:
    """
    Extract the video ID from a YouTube URL.
    """
    if "watch?v=" in youtube_url:
        return youtube_url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError(f"Unrecognized YouTube URL format: {youtube_url}")


def get_transcript(video_id: str) -> Optional[str]:
    """
    Retrieve the transcript of a YouTube video (if available).
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([seg['text'] for seg in transcript_list])
        return transcript_text
    except Exception as e:
        print(f"Failed to fetch transcript for {video_id}: {e}")
        return None


def summarize_with_claude(text: str, claude_api_key: str, model: str = "claude-3-sonnet-20240229", max_tokens: int = 300) -> str:
    """
    Summarize text using Claude's API
    """
    client = Anthropic(api_key=claude_api_key)
    prompt = (
        "Please summarize the following text:\n\n"
        f"{text}\n\n"
        "Summary:"
    )

    # Using ChatCompletion with a single user message
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )

    summary = response.content[0].text.strip()
    return summary


def cross_reference_summaries(summaries: List[str], claude_api_key: str, model: str = "claude-3-sonnet-20240229") -> str:
    """
    Use Chroma DB to find the top 3 summaries closest to the 'average' embedding.
    Then re-summarize them with Claude to create a combined summary.
    """

    # 1. Create an in-memory Chroma client
    client = chromadb.Client()

    # 2. Create a collection for storing summaries
    collection = client.get_or_create_collection("summaries_collection")

    # 3. Generate embeddings for each summary and insert into Chroma
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for idx, summary in enumerate(summaries):
        emb = embedding_model.encode(summary).tolist()  # convert to list for Chroma
        collection.add(
            documents=[summary],
            embeddings=[emb],
            ids=[f"summary-{idx}"],
        )

    # 4. Compute the average embedding
    all_embeddings = embedding_model.encode(summaries)
    avg_embedding = all_embeddings.mean(axis=0).tolist()

    # 5. Query the top 3 closest summaries to the average
    query_results = collection.query(
        query_embeddings=[avg_embedding],
        n_results=3  # top 3
    )

    # query_results is a dict with keys 'ids', 'embeddings', 'documents', 'metadatas'
    top_documents = query_results["documents"][0]  # top docs for the first (and only) query

    # 6. Re-summarize the combined text
    combined_text = "\n".join(top_documents)
    messages = [
        {
            "role": "user",
            "content": (
                "Below are multiple summaries of different videos:\n\n"
                f"{combined_text}\n\n"
                "Please create a combined summary that captures overlapping themes "
                "and major differences."
            )
        }
    ]

    claude_client = Anthropic(api_key=claude_api_key)
    response = claude_client.messages.create(
        model=model,
        system="You are a helpful AI assistant. Combine multiple video summaries into a single cohesive summary.",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    print(response)
    try:
        # Usually, Claude's reply is the last message with role="assistant"
        assistant_msg = response.content[0].text.strip()
        combined_summary = assistant_msg.strip()
    except (IndexError, KeyError):
        # Fallback if the structure isn't as expected
        combined_summary = "No response from Claude."

    return combined_summary


def topic_filter(text: str, topic: str) -> str:
    """
    Filter the text to focus on sentences containing the topic (case-insensitive).
    If none found, return the original text.
    """
    sentences = [s.strip() for s in text.split('.') if topic.lower() in s.lower()]
    filtered_text = '. '.join(sentences)
    return filtered_text if filtered_text else text


def process_videos(
    youtube_urls: List[str],
    claude_api_key: str,
    topic: Optional[str] = None,
    cross_reference: bool = False
) -> dict:
    """
    Core function that:
    1. Fetches transcripts
    2. Filters by topic if provided
    3. Summarizes using Claude
    4. Optionally cross-references summaries
    Returns a dict with 'summaries' and 'combined_summary'.
    """

    transcripts = []
    for url in youtube_urls:
        vid_id = extract_video_id(url)
        txt = get_transcript(vid_id)
        if txt:
            transcripts.append(txt)

    # Filter transcripts by topic
    if topic:
        transcripts = [topic_filter(t, topic) for t in transcripts]

    # Summarize each transcript
    summaries = []
    for txt in transcripts:
        summary = summarize_with_claude(txt, claude_api_key)
        summaries.append(summary)

    # Optionally cross reference
    combined_summary = None
    if cross_reference and len(summaries) > 1:
        combined_summary = cross_reference_summaries(summaries, claude_api_key)

    return {
        "summaries": summaries,
        "combined_summary": combined_summary
    }


#######################
# 1. FastAPI (REST API)
#######################
app = FastAPI()

class RequestModel(BaseModel):
    youtube_urls: List[str]
    claude_api_key: str
    topic: Optional[str] = None
    cross_reference: bool = False

@app.post("/process_videos")
def api_process_videos(data: RequestModel):
    """
    POST /process_videos
    Example Request Body:
    {
        "youtube_urls": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        "claude_api_key": "sk-...",
        "topic": "music",
        "cross_reference": true
    }
    """
    result = process_videos(
        youtube_urls= data.youtube_urls,
        claude_api_key= data.claude_api_key,
        topic= data.topic,
        cross_reference= data.cross_reference
    )
    return result