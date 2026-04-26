# ask.py

import hashlib
import os
import re
import sys
import warnings

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

DB_VERSION = "v2"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

llm = Ollama(model=os.getenv("OLLAMA_MODEL", "gemma4:e4b"))

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)


def get_db_path(url):
    key = f"{DB_VERSION}:{EMBEDDING_MODEL}:{url}"
    return f"./chroma_db_{hashlib.md5(key.encode()).hexdigest()}"


def normalize_url(url):
    url = url.strip()

    markdown_link = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", url)
    if markdown_link:
        return markdown_link.group(2)

    if not re.match(r"^https?://", url):
        raise ValueError(f"Invalid URL: {url}")

    return url


def fetch_html(url):
    response = requests.get(
        url,
        headers={"User-Agent": "crew-prj-rag/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    return response.text


def extract_infobox_qa(soup):
    infobox = soup.find("table", {"class": "infobox"})
    if not infobox:
        return ""

    lines = []
    for row in infobox.find_all("tr"):
        th = row.find("th")
        td = row.find("td")
        row_text = row.get_text(" ", strip=True)

        if not th or not td:
            if "Incumbent" in row_text:
                lines.append(row_text)
            continue

        key = th.get_text(" ", strip=True)
        value = td.get_text(" ", strip=True)
        if key and value:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def extract_clean_text(html):
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "sup"]):
        tag.decompose()

    parts = []

    title = soup.find("h1")
    if title:
        parts.append(f"Title: {title.get_text(' ', strip=True)}")

    infobox_text = extract_infobox_qa(soup)
    if infobox_text.strip():
        parts.append("[Infobox]\n" + infobox_text)

    for tag in soup(["table", "nav"]):
        tag.decompose()

    content = soup.find("div", {"id": "mw-content-text"})
    if content:
        parts.append(content.get_text("\n"))

    text = "\n\n".join(parts)
    if not text.strip():
        text = soup.get_text("\n")

    return text


def build_vectorstore(url, db_path):
    print("[초기 구축] 문서 수집 및 벡터화 진행...")

    html = fetch_html(url)
    clean_text = extract_clean_text(html)

    if not clean_text.strip():
        raise ValueError("텍스트 추출 실패")

    docs = [Document(page_content=clean_text, metadata={"source": url})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    splits = splitter.split_documents(docs)

    if not splits:
        raise ValueError("split 실패")

    Chroma.from_documents(
        documents=[d for d in splits if d.page_content.strip()],
        embedding=embeddings,
        persist_directory=db_path,
    )

    print(f"[완료] Vector DB 저장됨: {db_path}")


def load_vectorstore(db_path):
    print("[로딩] 기존 Vector DB 사용")
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
    )


def answer(question):
    print("[일반 AI 답변 처리 중...]")
    return llm.invoke(question)


def answer_with_context(question, vectorstore):
    print("[RAG 검색 수행 중...]")

    docs = vectorstore.similarity_search(question, k=7)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer the question based only on the context below.
Answer in the same language as the question.
If the answer is not in the context, say "모르겠습니다".

[Context]
{context}

[Question]
{question}

[Answer]
"""

    return llm.invoke(prompt)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('python ask.py "질문" "URL"')
        print('예시: python ask.py "미국의 현직 대통령은 누구인가요?" "https://en.wikipedia.org/wiki/President_of_the_United_States"\n')
        sys.exit(1)

    question = sys.argv[1]
    url = normalize_url(sys.argv[2])
    db_path = get_db_path(url)

    print("\n" + "=" * 70)
    print("AI 지식 검색 테스트 (RAG)")
    print("=" * 70)

    print("\n[결과 1] 일반 AI 답변:")
    print("-" * 50)
    print(answer(question))

    print("\n" + "=" * 70)
    print("\n[결과 2] RAG 적용 AI 답변:")
    print("-" * 50)

    if not os.path.exists(db_path):
        build_vectorstore(url, db_path)

    vs = load_vectorstore(db_path)
    print(answer_with_context(question, vs))

    print("\n" + "=" * 70 + "\n")
