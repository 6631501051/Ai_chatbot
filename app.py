import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- UI Setup ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 AI Movie Recommender")
st.write("Tell me what kind of vibe you want, and I'll recommend a movie!")

# --- Fetch API Token from Streamlit Secrets ---
hf_token = st.secrets.get("HF_TOKEN", "")
if not hf_token:
    st.error("Please add your HF_TOKEN to Streamlit Secrets!")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

@st.cache_resource
def setup_rag_pipeline():
    # 1. Create the MOVIE dataset file dynamically
    file_name = "movies_data.txt"
    if not os.path.exists(file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("""Name: Inception
Genre: Sci-Fi, Action, Thriller
Summary: A skilled thief is given a chance at redemption if he can successfully perform an inception—planting an idea into a target's subconscious.
Vibe: Mind-bending, intense, complex.
---
Name: Spirited Away
Genre: Animation, Fantasy, Adventure
Summary: A sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts.
Vibe: Magical, beautiful, emotional.
---
Name: The Grand Budapest Hotel
Genre: Comedy, Drama
Summary: A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel's glorious years.
Vibe: Quirky, visually stunning, funny.
---
Name: Interstellar
Genre: Sci-Fi, Drama
Summary: A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.
Vibe: Epic, emotional, thought-provoking.
""")

    # 2. Load & Split Data
    loader = TextLoader(file_name, encoding="utf-8")
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50).split_documents(docs)
    
    # 3. Create Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 4. Connect to OpenThaiGPT via API 
    llm = HuggingFaceEndpoint(
        repo_id="openthaigpt/openthaigpt-1.0.0-7b-chat",
        task="text-generation",
        max_new_tokens=250,
        temperature=0.7
    )

    # 5. Create the RAG Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI movie expert. Use the following context to recommend movies based on the user's request. Context: {context}"),
        ("human", "{input}")
    ])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

# --- Initialize Chat ---
rag_chain = setup_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! What kind of movie are you in the mood for today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Handle User Input ---
if user_input := st.chat_input("E.g., I want to watch something mind-bending..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with st.spinner("Searching the database..."):
        try:
            response = rag_chain.invoke({"input": user_input})
            answer = response["answer"]
        except Exception as e:
            answer = f"API Error: {e}"
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
