import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
api_key = os.getenv("API_KEY")

# Iluasjad: pealkiri, alapealkiri
st.title("🎓 AI Kursuse Nõustaja")
st.caption("Lihtne vestlusliides automaatvastusega.")

# Kontrolli, kas API võti on seatud
if not api_key:
    st.error("⚠️ API võti pole seadistatud!")
    st.stop()
    
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("data/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("data/puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df
embedder, df, embeddings_df = get_models()

# Tervituse sõnum
with st.chat_message("assistant"):
    st.markdown("👋 Tere tulemast! Kirjelda altpoolt, mida soovid õppida ja teen sulle kursuste ettepanekuid. Ma otsin sulle kõige sobivamaid kursusi.")

# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Korjame üles uue kasutaja sisendi (Action)
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    # Kuvame kohe kasutaja sõnumi ja salvestame selle ka ajalukku
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Genereerime vastuse
    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "❌ API võti pole seatud. Palun kontrolli .env faili!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # UUS ÜLESANNE: Filtreerimine enne semantilist otsingut
            with st.spinner("Otsin sobivaid kursusi..."):
                # 1. ühenda kaks andmetabelit ja filtreeri esmalt EAP-de ja semestri alusel
                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                mask = ((merged_df['semester'] == 'kevad' )& (merged_df["eap"]==6))
                filtered_df = merged_df[mask].copy()

                #kontroll (sanity check)
                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                else:
                    # Arvutame sarnasuse ja sorteerime tabeli
                    query_vec = embedder.encode([prompt])[0]
                    # lisa embedding andmefreimile score rida
                    embeddings_array = np.vstack(filtered_df['embedding'].values)
                    filtered_df['score'] = cosine_similarity([query_vec], embeddings_array)[0]
                    
                    # Leiame 5 kõige sarnasemat (suurim skoor)
                    return_N = 5
                    results_df = filtered_df.sort_values('score', ascending=False).head(return_N)
                    results_df.drop(['score', 'embedding'], axis=1, inplace=True)
                    context_text = results_df.to_string()

                # 3. LLM vastus
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                system_prompt = {
                    "role": "system", 
                    "content": f"Oled nõustaja. Kasuta järgmisi kursusi (filtreeritud: inglise keel):\n\n{context_text}"
                }
                
                messages_to_send = [system_prompt] + st.session_state.messages
                
                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it",
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Viga: {e}")