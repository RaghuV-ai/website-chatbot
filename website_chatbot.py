pip install beautifulsoup4
import streamlit as st
import os
import re
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag, urlparse
import urllib.robotparser as robotparser
from collections import deque

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# -----------------------
# Configuration & Secrets
# -----------------------
# Expect OPENAI_API_KEY in Streamlit secrets for deployment (Streamlit Cloud/GitHub)
# Add this to .streamlit/secrets.toml:
# OPENAI_API_KEY = "sk-..."
os.environ["OPENAI_API_KEY"] = st.secrets["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"]

st.set_page_config(page_title="Chat with Websites", page_icon="🌐")
st.title("Chat with your Websites 🌐")

# -----------------------
# Session State
# -----------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = False
if "indexed_pages" not in st.session_state:
    st.session_state.indexed_pages = []  # list of (url, chars)

# -----------------------
# Helpers
# -----------------------
USER_AGENT = "Mozilla/5.0 (compatible; WebsiteChatbot/1.0; +https://example.com/bot)"

def normalize_url(base_url: str, link: str) -> str:
    """Resolve relative links and strip fragments."""
    try:
        href = urljoin(base_url, link)
        href, _ = urldefrag(href)
        return href
    except Exception:
        return ""

def is_html_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return "text/html" in ctype or "application/xhtml+xml" in ctype

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove script/style/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    # optional: remove nav/footer/aside
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.extract()
    # get text
    text = soup.get_text(separator="\n")
    # collapse excessive newlines
    text = re.sub(r"\n{2,}", "\n", text)
    return clean_text(text)

def can_fetch(url: str) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # if robots fails, be conservative but allow
        return True

def same_domain(u1: str, u2: str) -> bool:
    return urlparse(u1).netloc == urlparse(u2).netloc

def crawl(start_urls, max_pages=20, same_domain_only=True, timeout=10):
    visited = set()
    texts_by_url = {}
    q = deque()

    # seed queue
    for u in start_urls:
        if not u.startswith("http" ):
            u = "https://" + u
        q.append(u)

    while q and len(visited) < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        # robots.txt check
        if not can_fetch(url):
            continue

        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except Exception:
            continue

        if resp.status_code != 200 or not is_html_response(resp):
            continue

        text = extract_visible_text(resp.text)
        if len(text) < 200:  # skip tiny pages
            continue
        texts_by_url[url] = text

        # enqueue links
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = normalize_url(url, a["href"])
            if not href or href in visited:
                continue
            if same_domain_only and not same_domain(start_urls[0], href):
                continue
            if href.startswith("mailto:") or href.startswith("tel:"):
                continue
            if any(href.lower().endswith(ext) for ext in [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".rar", ".7z", ".mp4", ".mp3"]):
                continue
            q.append(href)

    return texts_by_url

def split_into_chunks(texts_by_url, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    all_chunks = []
    metadata = []
    for url, text in texts_by_url.items():
        chunks = splitter.split_text(text)
        for ch in chunks:
            all_chunks.append(ch)
            metadata.append({"source": url})
    return all_chunks, metadata

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o")

    template = """You are a helpful AI assistant that answers questions about the crawled website content.
Use the context snippets to answer the user's question. If you are unsure or the answer is not in the context, say you don't know.
Always cite the most relevant source URL(s) you used from the context in your answer when possible.

{context}

Question: {question}
Helpful answer (include source URL if relevant):"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return conversation_chain

def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def process_websites(urls, max_pages, same_domain_only):
    try:
        with st.spinner("Crawling and indexing websites..."):
            pages = crawl(urls, max_pages=max_pages, same_domain_only=same_domain_only)
            if not pages:
                st.error("No crawlable text content found. Please check the URLs or increase Max pages.")
                return False
            st.session_state.indexed_pages = [(u, len(t)) for u, t in pages.items()]

            chunks, metadata = split_into_chunks(pages)
            if not chunks:
                st.error("No content chunks produced.")
                return False

            vectorstore = build_vectorstore(chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.session_state.processComplete = True
            return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False

# -----------------------
# Sidebar: Input & Controls
# -----------------------
with st.sidebar:
    st.subheader("Your Websites")
    st.write("Enter one or more URLs (one per line). The crawler will stay on the first domain if 'Same domain only' is checked.")
    urls_text = st.text_area("URLs", placeholder="https://example.com\nhttps://docs.example.com/guide")
    col1, col2 = st.columns(2)
    with col1:
        max_pages = st.number_input("Max pages", min_value=1, max_value=200, value=20, step=1)
    with col2:
        same_domain_only = st.checkbox("Same domain only", value=True)

    if st.button("Process") and urls_text.strip():
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        success = process_websites(urls, int(max_pages), same_domain_only)
        if success:
            st.success("Processing complete!")

# -----------------------
# Main Chat Interface
# -----------------------
if st.session_state.processComplete:
    with st.expander("Indexed pages (URL • characters)", expanded=False):
        for u, n in sorted(st.session_state.indexed_pages):
            st.write(f"{u} • {n}")

    user_question = st.chat_input("Ask a question about your websites:")
    if user_question:
        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": user_question})
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response["answer"]))
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
else:
    st.write("👈 Enter website URLs in the sidebar and click **Process** to get started!")

# -----------------------
# Notes for GitHub Deployment
# -----------------------
# 1) Create requirements.txt with (minimum):
#    streamlit
#    requests
#    beautifulsoup4
#    langchain
#    langchain-openai
#    langchain-community
#    faiss-cpu
#    tiktoken
#    PyYAML
# 2) Add .streamlit/secrets.toml with your OpenAI key.
# 3) Run locally:  streamlit run website_chatbot.py
# -----------------------
