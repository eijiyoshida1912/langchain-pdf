import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

api_key = st.secrets["OPENAI_API_KEY"]

# 最初のほうに書いておくと安全！
if "query" not in st.session_state:
    st.session_state["query"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# LangChainオブジェクトに渡す
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
chat = ChatOpenAI(openai_api_key=api_key)

st.title("PDF QA チャット")

uploaded_file = st.file_uploader("PDFをアップロード", type=["pdf"])

# 要約プロンプト（日本語指定）
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="以下の文章を日本語で要約してください。\n{text}"
)

if uploaded_file:
    # ファイル名（または bytes の hash など）で変化を検知
    file_key = uploaded_file.name  # ←または hash(uploaded_file.read()) を使う手もある

    # chat履歴を初期化（ファイルが変わったらリセットもここで）
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ファイルが前と違うならリセット
    if st.session_state.get("uploaded_file_key") != file_key:
        st.session_state["uploaded_file_key"] = file_key
        if "summary" in st.session_state:
            del st.session_state["summary"]
        if "split_docs" in st.session_state:
            del st.session_state["split_docs"]
        if "db" in st.session_state:
            del st.session_state["db"]
        st.session_state["query"] = ""
        st.session_state["chat_history"] = []


    if "summary" not in st.session_state:
        # 一時保存
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # ＜NEW: 視覚的にわかる進行パネル＞
        with st.status("📥 PDFを解析しています…", expanded=True) as status:
            status.write("PDFを読み込んでいます…")
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
    
            status.write("テキストを分割しています…")
            splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
            split_docs = splitter.split_documents(docs)
    
            status.write("要約を生成しています…")
            summary_chain = load_summarize_chain(
                llm=chat,
                chain_type="map_reduce",
                map_prompt=prompt_template,
                combine_prompt=prompt_template
            )
            summary = summary_chain.run(split_docs)
    
            status.update(label="✅ 要約が完了しました", state="complete")

        # セッションに保存
        st.session_state["split_docs"] = split_docs
        st.session_state["summary"] = summary
        st.session_state["db"] = FAISS.from_documents(split_docs, embeddings)

        # ＜NEW: 右下ポップアップ通知＞
        st.toast("PDFの要約が完了しました", icon="✅")

    # 表示だけ再利用
    st.subheader("📄 PDFの要約")
    st.info(st.session_state["summary"])


    # --- チャット履歴の管理（role: "user" / "assistant"） ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # 既存履歴を先に描画（←これがあるから即時に見える）
    for m in st.session_state["messages"]:
        st.chat_message(m["role"]).markdown(m["content"])
    
    # 入力欄
    query = st.chat_input("質問をどうぞ")


    # 送信後の処理
    if query:
        # 1) ユーザー発言は即表示
        st.session_state["messages"].append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
    
        # 2) アシスタントの枠だけ先に用意
        assistant_box = st.chat_message("assistant")
    
        # 3) ステータスを入れるプレースホルダ
        ph = st.empty()
        with ph.container():
            with st.status("🔎 回答を生成中…", expanded=True) as status:
                status.write("関連箇所を検索しています…")
                db = st.session_state["db"]
                results = db.similarity_search(query, k=3)
    
                content = "\n".join([d.page_content for d in results])
                prompt = PromptTemplate(
                    template="""
    以下の文章を参考にして、質問に日本語で答えてください。
    
    文章:
    {document}
    
    質問: {query}
    """,
                    input_variables=["document", "query"]
                )
    
                status.write("回答を作成しています…")
                chat = ChatOpenAI(openai_api_key=api_key)
                response = chat([HumanMessage(content=prompt.format(document=content, query=query))])
    
                status.update(label="✅ 回答ができました", state="complete", expanded=False)
    
        # 4) ステータスを消す
        ph.empty()
    
        # 5) 回答描画＆履歴保存
        assistant_box.markdown(response.content)
        st.session_state["messages"].append({"role": "assistant", "content": response.content})