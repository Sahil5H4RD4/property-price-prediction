"""
RAG Vectorstore Builder
=======================
Scrapes Wikipedia articles about Indian real estate,
chunks them, embeds with HuggingFace, and saves a FAISS index.
"""

import logging
import os
import time

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds between retries

WIKIPEDIA_SOURCES = [
    (
        "https://en.wikipedia.org/wiki/Real_Estate_(Regulation_and_Development)_Act,_2016",
        "RERA 2016 Act",
    ),
    (
        "https://en.wikipedia.org/wiki/Real_estate_in_India",
        "Real Estate Market Trends in India",
    ),
    (
        "https://en.wikipedia.org/wiki/Housing_in_India",
        "Housing in India (Demographics & Supply)",
    ),
]


def fetch_wikipedia_text(url: str) -> str:
    """Fetch and extract paragraph text from a Wikipedia page.

    Retries up to MAX_RETRIES times with exponential backoff.
    Raises requests.HTTPError on non-2xx responses.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; PropertyPriceBot/1.0)'}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("Fetching %s (attempt %d/%d)", url, attempt, MAX_RETRIES)
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except requests.Timeout:
            logger.warning("Timeout fetching %s (attempt %d)", url, attempt)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF * attempt)
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF * attempt)

    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = '\n\n'.join(p.get_text() for p in paragraphs if p.get_text(strip=True))
    logger.info("Fetched %d chars from %s", len(text), url)
    return text


def build_vectorstore() -> None:
    """Fetch articles, chunk, embed, and save FAISS vectorstore."""
    logger.info("Building RAG vectorstore from %d sources", len(WIKIPEDIA_SOURCES))

    documents = []
    for url, title in WIKIPEDIA_SOURCES:
        try:
            text = fetch_wikipedia_text(url)
            documents.append(
                Document(page_content=text, metadata={"source": url, "title": title})
            )
        except Exception as exc:
            logger.error("Failed to fetch '%s': %s", title, exc)

    if not documents:
        raise RuntimeError("No documents fetched — cannot build vectorstore.")

    logger.info("Splitting %d documents into chunks", len(documents))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    logger.info("Building FAISS index (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    vectorstore_path = os.path.join(base_dir, 'data', 'vectorstore')
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    logger.info("Vectorstore saved to %s", vectorstore_path)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    build_vectorstore()
