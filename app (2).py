
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. Load Data
loader = TextLoader('movies.txt')
splits = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(loader.load())

# 2. Vector Store
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

# 3. Model
model_id = 'openthaigpt/openthaigpt-1.0.0-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', load_in_4bit=True)
hf_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 4. Chain
prompt = ChatPromptTemplate.from_messages([('system', 'You are a Cinephile Movie Critic. Context: {context}'), ('human', '{input}')])
def format_docs(docs): return '

'.join(doc.page_content for doc in docs)

chain = ({'context': retriever | format_docs, 'input': RunnablePassthrough()} | prompt | llm | StrOutputParser())

# 5. Interface
demo = gr.ChatInterface(fn=lambda m, h: chain.invoke(m), title='Cinephile Radar')
if __name__ == '__main__':
    demo.launch()
