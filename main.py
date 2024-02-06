import pysqlite3.dbapi2 as sqlite3
import sys
sys.modules['sqlite3'] = sqlite3

from pydantic import BaseModel
import openai # OpenAI KYE Import
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = FastAPI()

class Question(BaseModel):
    text: str

import os
openai.api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=500)

# ChromaDBの準備
db2 = Chroma(
    persist_directory='persist_directory',
    collection_name="langchain_store",
    embedding_function=embeddings,
)

# データベースからretriever作成
retriever = db2.as_retriever(search_kwargs={"k": 3}) # Topkもここの引数で指定できる

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.post("/answer")
def get_answer(question: Question):
    # ここでモデルに質問を渡し、応答を取得するロジックを追加
    query = question.text
    answer = qa.invoke(query)
    return {"answer": answer}