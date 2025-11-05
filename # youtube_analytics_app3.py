# youtube_analytics_app.py — Enhanced version with clickable video links and stats

from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import os, tempfile, re, subprocess

# LangChain components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from langchain_community.llms import Ollama
    OLLAMA_IMPORTED = True
except Exception:
    OLLAMA_IMPORTED = False


# ==================== Streamlit Setup ====================
st.set_page_config(page_title="YouTube Channel AI Dashboard", layout="wide")
st.title("📊 YouTube Channel Analytics + Smart Search (Free)")
st.markdown("Keyword-aware semantic search with optional offline summarization via Ollama Mistral.")


# ==================== Session State ====================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None


# ==================== Utility Functions ====================
def ollama_available() -> bool:
    if not OLLAMA_IMPORTED:
        return False
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def get_channel_videos_df(api_key: str, channel_name: str, max_results: int = 50, order: str = "date") -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=api_key)
    ch_resp = youtube.search().list(part="snippet", q=channel_name, type="channel", maxResults=1).execute()
    if not ch_resp.get("items"):
        st.error(f"No channel found for '{channel_name}'.")
        return pd.DataFrame()

    channel_id = ch_resp["items"][0]["id"]["channelId"]
    v_resp = youtube.search().list(part="snippet", channelId=channel_id, type="video", order=order, maxResults=max_results).execute()
    video_ids = [i["id"]["videoId"] for i in v_resp.get("items", [])]

    if not video_ids:
        return pd.DataFrame()

    s_resp = youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(video_ids)).execute()
    data = []
    for item in s_resp.get("items", []):
        sn = item["snippet"]
        stt = item.get("statistics", {})
        det = item.get("contentDetails", {})
        data.append({
            "Title": sn.get("title"),
            "Published": pd.to_datetime(sn.get("publishedAt")),
            "Description": sn.get("description", ""),
            "Video_ID": item["id"],
            "URL": f"https://www.youtube.com/watch?v={item['id']}",
            "Duration": det.get("duration", ""),
            "Views": int(stt.get("viewCount", 0)),
            "Likes": int(stt.get("likeCount", 0)),
            "Comments": int(stt.get("commentCount", 0))
        })
    return pd.DataFrame(data)


def plot_trend_over_time(df, metric="Views"):
    if df.empty:
        return
    df2 = df.sort_values("Published")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df2["Published"], df2[metric], marker="o", linewidth=2, color="tab:blue")
    ax.set_title(f"{metric} Trend Over Time", fontsize=14, weight="bold")
    ax.set_xlabel("Published Date")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.5)
    plt.xticks(rotation=30)
    st.pyplot(fig)


def plot_top_videos(df, metric="Views", top_n=10):
    if df.empty:
        return
    df2 = df.sort_values(metric, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df2["Title"], df2[metric], color="skyblue", edgecolor="black")
    ax.set_title(f"Top {top_n} Videos by {metric}", fontsize=14, weight="bold")
    ax.set_xlabel(metric)
    ax.set_ylabel("Video Title")
    ax.invert_yaxis()
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f"{int(bar.get_width()):,}", va="center")
    plt.tight_layout()
    st.pyplot(fig)


STOP = set("a an and are as at be but by for from has have i in is it its of on or that the this to was were will with you your we our".split())
def extract_keywords(q):
    phrases = re.findall(r'"([^"]+)"', q)
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", q) if len(w) > 3 and w.lower() not in STOP]
    return list(set([w.lower() for w in words + phrases]))


def keyword_filter_indices(df, keywords):
    if df.empty or not keywords:
        return []
    mask = False
    for kw in keywords:
        m = df["Title"].str.contains(kw, case=False, na=False) | df["Description"].str.contains(kw, case=False, na=False)
        mask = m if isinstance(mask, bool) and not mask else (mask | m)
    return df[mask].index.astype(int).tolist()


def build_vector_db(df):
    if df.empty:
        return None
    docs, metas = [], []
    for i, r in df.reset_index(drop=True).iterrows():
        text = (
            f"Title: {r['Title']}\n"
            f"Description: {r['Description']}\n"
            f"Views: {r['Views']} Likes: {r['Likes']} Comments: {r['Comments']}"
        )
        docs.append(text)
        metas.append({"row": int(i), "title": r["Title"], "views": int(r["Views"]), "url": r["URL"]})
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(docs, embedding=embeddings, metadatas=metas)
    return vectordb


def summarize_with_ollama(question, hits):
    context = "\n\n".join([f"Title: {t}\nSnippet: {s}" for t, s, _ in hits])
    prompt = (
        f"You are an assistant summarizing YouTube videos.\n\n"
        f"Question: {question}\n\n"
        f"Relevant videos:\n{context}\n\n"
        "Summarize and list the most relevant videos with bullet points and short explanations."
    )
    llm = Ollama(model="mistral", temperature=0.2)
    return llm(prompt)


def chat_with_channel(df, vectordb, question, use_kw=True, use_sum=True):
    if vectordb is None or df.empty:
        return "Please fetch data and build knowledge base first."

    kw = extract_keywords(question) if use_kw else []
    idx = keyword_filter_indices(df, kw) if kw else []
    filt = {"row": {"$in": idx}} if idx else None

    docs = vectordb.similarity_search(question, k=5, filter=filt)
    if not docs:
        return "No relevant results found."

    hits = []
    for d in docs:
        title = d.metadata["title"]
        snippet = d.page_content[:300].replace("\n", " ").strip()
        url = d.metadata.get("url", "")
        hits.append((title, snippet, url))

    if use_sum and ollama_available():
        try:
            return summarize_with_ollama(question, hits)
        except Exception:
            pass

    # clickable results
    formatted = "🔍 **Top Related Videos:**\n\n"
    for t, s, u in hits:
        stats_row = df[df["URL"] == u]
        if not stats_row.empty:
            views = int(stats_row["Views"].iloc[0])
            likes = int(stats_row["Likes"].iloc[0])
            comments = int(stats_row["Comments"].iloc[0])
            formatted += f"🎥 **[{t}]({u})**  \n👁️ {views:,} views | 👍 {likes:,} likes | 💬 {comments:,} comments  \n{s}\n\n"
        else:
            formatted += f"🎥 **[{t}]({u})**  \n{s}\n\n"
    return formatted


# ==================== Sidebar Controls ====================
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("YouTube API Key", type="password")
channel_name = st.sidebar.text_input("Channel Name", value="Ash ki Baat")
max_results = st.sidebar.slider("Number of Videos", 10, 50, 50)
order = st.sidebar.selectbox("Order By", ["date", "viewCount", "rating", "relevance"])

if st.sidebar.button("🚀 Fetch Channel Data"):
    if not api_key or not channel_name:
        st.warning("Please enter your YouTube API key and channel name.")
    else:
        with st.spinner("Fetching channel data..."):
            df = get_channel_videos_df(api_key, channel_name, max_results=max_results, order=order)
        if not df.empty:
            st.session_state.df = df
            st.success(f"Fetched {len(df)} videos from '{channel_name}'.")
        else:
            st.error("No data found.")


# ==================== Dashboard ====================
if not st.session_state.df.empty:
    df = st.session_state.df
    st.dataframe(df[["Title", "Published", "Views", "Likes", "Comments"]])

    st.markdown("### 📈 Channel Visualizations")
    metric = st.selectbox("Metric", ["Views", "Likes", "Comments"])
    top_n = st.slider("Top N Videos", 5, 20, 10)
    c1, c2 = st.columns(2)
    with c1: plot_top_videos(df, metric, top_n)
    with c2: plot_trend_over_time(df, metric)

    st.markdown("### 🧠 Chat with the Channel (Offline)")
    use_kw = st.checkbox("Use Keyword/Entity Filter", True)
    use_sum = st.checkbox("Use Local Summarizer (Ollama)", True)
    st.write("Ollama Available:", "✅" if ollama_available() else "❌")

    if st.button("Build Channel Knowledge Base"):
        with st.spinner("Building local embeddings..."):
            vectordb = build_vector_db(df)
        if vectordb:
            st.session_state.vectordb = vectordb
            st.success("Knowledge base created successfully!")

    if st.session_state.vectordb:
        query = st.text_area("Ask about the channel:", "Videos related to Virat Kohli")
        if st.button("Ask"):
            with st.spinner("Searching and analyzing..."):
                answer = chat_with_channel(df, st.session_state.vectordb, query, use_kw, use_sum)
            st.markdown(answer)
