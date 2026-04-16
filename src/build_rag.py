import os
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

def fetch_wikipedia_text(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text from paragraphs
    paragraphs = soup.find_all('p')
    text = '\n\n'.join([p.get_text() for p in paragraphs])
    return text

def main():
    print("Fetching target real estate data...")
    urls = [
        ("https://en.wikipedia.org/wiki/Real_Estate_(Regulation_and_Development)_Act,_2016", "RERA 2016 Act"),
        ("https://en.wikipedia.org/wiki/Real_estate_in_India", "Real Estate Market Trends in India"),
        ("https://en.wikipedia.org/wiki/Housing_in_India", "Housing in India (Demographics & Supply)")
    ]
    
    documents = []
    for url, title in urls:
        print(f"Fetching {title}...")
        try:
            text = fetch_wikipedia_text(url)
            documents.append(Document(page_content=text, metadata={"source": url, "title": title}))
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            
    if not documents:
        print("No documents fetched. Exiting...")
        return
        
    print("Splitting texts...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    print(f"Creating FAISS index with {len(docs)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    vectorstore_path = os.path.join(base_dir, 'data', 'vectorstore')
    
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    print(f"Vectorstore successfully saved to {vectorstore_path}")

if __name__ == '__main__':
    main()
