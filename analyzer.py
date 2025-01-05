import os
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoModel, AutoTokenizer, AutoModelEmbeddings  # For embeddings
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import concurrent.futures
from typing import List, Optional

# Set your OpenAI API key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")