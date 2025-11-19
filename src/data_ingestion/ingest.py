import os
import json
import time
import argparse
import fitz
import ebooklib
import requests
from ebooklib import epub
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


JSON_DIR = "data/jsons"
BOOKS_DIR = "data/source_books"
OUTPUT_JSON_CHUNKS = "data/jsons/processed_chunks.json"


SITE_CONFIGS = {
    "python_reference": {
        "base_url": "https://docs.python.org/3/reference/index.html",
        "content_selector": 'div[role="main"]',
        "link_selector": 'div[role="main"]',
        "link_filter": lambda href: href.endswith('.html') and '#' not in href and 'genindex' not in href
    },
    "python_stdlib": {
        "base_url": "https://docs.python.org/3/library/index.html",
        "content_selector": 'div[role="main"]',
        "link_selector": 'div[role="main"]',
        "link_filter": lambda href: href.endswith('.html') and '#' not in href and 'genindex' not in href
    },
    "fastapi": {
        "base_url": "https://fastapi.tiangolo.com/tutorial/",
        "content_selector": "main",
        "link_selector": "nav.md-nav--primary",
        "link_filter": lambda href: not href.startswith(('.', '#', 'http')) and href.endswith('/')
    },
    "pydantic": {
        "base_url": "https://docs.pydantic.dev/latest/",
        "content_selector": "main",
        "link_selector": "nav.md-nav--primary",
        "link_filter": lambda href: not href.startswith('http') and not href.startswith('#') and href.endswith('/')
    },
    "sqlalchemy": {
        "base_url": "https://docs.sqlalchemy.org/en/20/tutorial/index.html",
        "content_selector": 'div[role="main"]',
        "link_selector": 'div#docs-sidebar-inner',
        "link_filter": lambda href: not href.startswith(('#', 'http', '//')) and href.endswith('.html')
    },
    "seaborn": {
        "base_url": "https://seaborn.pydata.org/api.html",
        "content_selector": 'article',
        "link_selector": 'div.bd-sidebar-primary',
        "link_filter": lambda href: href.startswith('generated/') and href.endswith('.html')
    },
    "plotly": {
        "base_url": "https://plotly.com/python/",
        "content_selector": "section.--page-body",
        "link_selector": "section.--page-body",
        "link_filter": lambda href: href.startswith('/python/') and len(href) > len('/python/')
    },
    "flask": {
        "base_url": "https://flask.palletsprojects.com/en/3.0.x/",
        "content_selector": 'div[role="main"]',
        "link_selector": 'div.toctree-wrapper',
        "link_filter": lambda href: not href.startswith(('http', '#')) and href.endswith('/')
    },
    "pandas": {
        "base_url": "https://pandas.pydata.org/docs/reference/index.html",
        "content_selector": 'article.bd-article',
        "link_selector": 'div.toctree-wrapper',
        "link_filter": lambda href: href.endswith('.html') and '#' not in href and 'api/' not in href
    },
}

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "llm-teacher-python-db"
VECTOR_SIZE = 1024
BATCH_SIZE = 32
MIN_CHUNK_LENGTH = 50


def scrape_and_chunk_page(
    url: str, 
    config: dict, 
    source_key: str, 
    splitter: RecursiveCharacterTextSplitter
) -> list[dict]:
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return []
    
    soup = BeautifulSoup(response.content, 'lxml')
    main_content = soup.select_one(config['content_selector'])
    if not main_content: return []

    text_pieces = [p.get_text(" ", strip=True) for p in main_content.find_all(['p', 'li', 'h1', 'h2', 'h3', 'code'])]
    full_text = " ".join(text_pieces)
    if not full_text.strip(): return []

    chunks_text = splitter.split_text(full_text)
    
    return [{
        "text": text,
        "metadata": {"source_type": "documentation", "source_name": source_key, "url": url}
    } for text in chunks_text]


def find_docs_links(site_key: str) -> list[str]:
    config = SITE_CONFIGS[site_key]
    base_url = config['base_url']
    try:
        response = requests.get(base_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
    except requests.RequestException: return []

    soup = BeautifulSoup(response.content, 'lxml')
    link_area = soup.select_one(config['link_selector'])
    if not link_area: return []

    found_links = set()
    for a_tag in link_area.find_all('a', href=True):
        href = a_tag['href']
        if config['link_filter'](href):
            found_links.add(urljoin(base_url, href))
            
    return sorted(list(found_links))


def extract_text_from_pdf(file_path: str) -> str:
    try:
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception: return ""


def extract_text_from_epub(file_path: str) -> str:
    try:
        book = epub.read_epub(file_path)
        full_text = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            full_text += soup.get_text() + "\n\n"
        return full_text
    except Exception: return ""


def process_sources(args, text_splitter) -> list[dict]:
    all_chunks = []

    if args.from_json:
        print("[ingest] from json")
        for filename in args.from_json:
            filepath = os.path.join(JSON_DIR, filename)
            if not os.path.exists(filepath):
                print(f"[ingest] !!!WARNING!!! file '{filepath}' not found. Skipping.")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                        all_chunks.extend(data)
                        print(f"[ingest] loaded {len(data)} chunks from '{filepath}'")
                    else:
                        print(f"[ingest] !!!WARNING!!! file '{filepath}' is not a list of dicts. Skipping.")
            except json.JSONDecodeError:
                print(f"[ingest] !!!WARNING!!! file '{filepath}' is not a valid JSON. Skipping.")
        return all_chunks

    if args.books:
        print(f"[ingest] reading books from: {BOOKS_DIR}")
        for filename in os.listdir(BOOKS_DIR):
            file_path = os.path.join(BOOKS_DIR, filename)
            full_text = ""
            if filename.lower().endswith('.pdf'):
                full_text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith('.epub'):
                full_text = extract_text_from_epub(file_path)
            
            if full_text:
                chunks = text_splitter.split_text(full_text)
                for chunk_text in chunks:
                    all_chunks.append({
                        "text": chunk_text,
                        "metadata": {"source_type": "book", "source_name": filename, "url": None}
                    })
                print(f"[ingest] file '{filename}', got {len(chunks)} chunks.")

    if args.docs:
        for site_key in args.docs:
            if site_key not in SITE_CONFIGS:
                print(f"[ingest] !!!WARNING!!! no config for '{site_key}'. Skipping.")
                continue
            links = find_docs_links(site_key)
            print(f"[ingest] found {len(links)} links.")
            for url in tqdm(links, desc=f"parsing {site_key}"):
                page_chunks = scrape_and_chunk_page(url, SITE_CONFIGS[site_key], site_key, text_splitter)
                all_chunks.extend(page_chunks)
                time.sleep(0.2)
    
    return all_chunks


def index_to_qdrant(documents: list[dict]):
    print("\n[ingest] Indexing to Qdrant")
    
    if not documents:
        print("[ingest] !!!WARNING!!! no documents to index")
        return

    initial_count = len(documents)
    documents = [doc for doc in documents if len(doc.get('text', '')) > MIN_CHUNK_LENGTH]
    print(f"[ingest] after filtering len > {MIN_CHUNK_LENGTH} chars, got {len(documents)} chunks.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[ingest] embedder: {device.upper()}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    print(f"[ingest] Collection '{COLLECTION_NAME}' created/recreated.")
    
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Loading to Qdrant"):
        batch = documents[i:i + BATCH_SIZE]
        texts_to_encode = [doc['text'] for doc in batch]
        
        texts_with_instruction = [f'passage: {text}' for text in texts_to_encode]
        embeddings = embedding_model.encode(texts_with_instruction, normalize_embeddings=True)
        
        points = [
            models.PointStruct(
                id=i + idx,
                vector=embeddings[idx].tolist(),
                payload=doc
            ) for idx, doc in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=False)

    print(f"[ingest] Now {client.get_collection(collection_name=COLLECTION_NAME).points_count} vectors in '{COLLECTION_NAME}'.")


def main():
    parser = argparse.ArgumentParser(description="Пайплайн сбора и индексации данных для RAG.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--books', action='store_true', help="Парсить локальные книги.")
    source_group.add_argument('--docs', nargs='+', choices=SITE_CONFIGS.keys(), help="Парсить документацию по ключам.")
    source_group.add_argument('--from-json', nargs='+', help="Загрузить чанки из готовых JSON-файлов в папке 'data/jsons'.")
    parser.add_argument('--skip-indexing', action='store_true', help="Только собрать данные, без индексации в Qdrant.")
    args = parser.parse_args()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    all_chunks = process_sources(args, text_splitter)
    
    if not all_chunks:
        print("[ingest] !!!WARNING!!! parsed 0 chunks.")
        return
        
    print(f"\n[ingest] total parsed {len(all_chunks)} chunks.")
    
    if not args.from_json:
        os.makedirs(os.path.dirname(OUTPUT_JSON_CHUNKS), exist_ok=True)
        with open(OUTPUT_JSON_CHUNKS, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"[ingest] all chunks saved to {OUTPUT_JSON_CHUNKS}")

    if args.skip_indexing:
        print("[ingest] skip indexing.")
    else:
        index_to_qdrant(all_chunks)


if __name__ == "__main__":
    main()
