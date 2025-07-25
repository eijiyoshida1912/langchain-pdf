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

# ↓こうすれば読み取れる！
# api_key = os.getenv("OPENAI_API_KEY")
api_key = st.secrets["OPENAI_API_KEY"]

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

    # ファイルが前と違うならリセット
    if st.session_state.get("uploaded_file_key") != file_key:
        st.session_state.clear()  # 必要なキーだけ削除するなら del でもOK
        st.session_state["uploaded_file_key"] = file_key
        st.session_state["query"] = ""  # ← このキーの値をクリア

    if "summary" not in st.session_state:
        # 一時保存
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, separator="\n")
        split_docs = splitter.split_documents(docs)

        # 要約（日本語）
        summary_chain = load_summarize_chain(
            llm=chat,
            chain_type="map_reduce",
            map_prompt=prompt_template,
            combine_prompt=prompt_template
        )
        summary = summary_chain.run(split_docs)

        # セッションに保存
        st.session_state["split_docs"] = split_docs
        st.session_state["summary"] = summary
        st.session_state["db"] = FAISS.from_documents(split_docs, embeddings)

    # 表示だけ再利用
    st.subheader("📄 PDFの要約")
    st.info(st.session_state["summary"])

    # まだ作ってないときだけ（初回）
    if "db" not in st.session_state:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        db = FAISS.from_documents(split_docs, embeddings)
        st.session_state["db"] = db

    query = st.text_input("質問をどうぞ", key="query")
    if query:
        results = st.session_state["db"].similarity_search(query, k=3)

        content = "\n".join([d.page_content for d in results])
        prompt = PromptTemplate(template="""
文章:
{document}

質問: {query}
""", input_variables=["document", "query"])

        chat = ChatOpenAI()
        response = chat([
            HumanMessage(content=prompt.format(document=content, query=query))
        ])
        st.write(response.content)
