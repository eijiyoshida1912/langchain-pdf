import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import SpacyTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

# ↓こうすれば読み取れる！
api_key = os.getenv("OPENAI_API_KEY")

# LangChainオブジェクトに渡す
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
chat = ChatOpenAI(openai_api_key=api_key)

st.title("PDF QA チャット")

uploaded_file = st.file_uploader("PDFをアップロード", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    splitter = SpacyTextSplitter(chunk_size=300, pipeline="ja_core_news_sm")
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)

    query = st.text_input("質問をどうぞ")
    if query:
        results = db.similarity_search(query, k=3)

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
