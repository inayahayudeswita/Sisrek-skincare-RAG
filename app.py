import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import PyPDF2
import os
import re

# =====================================================
# 1. PAGE CONFIG & STYLE
# =====================================================
st.set_page_config(
    page_title="GlowAI - Skincare Expert",
    page_icon="✨",
    layout="centered"
)

st.markdown("""
<style>
.stApp { background-color:#ffffff; }
.main-header { text-align:center; margin-bottom:2rem; }
.main-header h1 { color:#008080; font-weight:800; }
.result-card {
    background:#f0fdf4;
    padding:20px;
    border-radius:10px;
    border-left: 5px solid #008080;
    margin-top:20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.result-card h3 { color: #008080; }
.warning-card {
    background:#fff3cd;
    padding:15px;
    border-radius:10px;
    border:1px solid #ffeeba;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. MODEL CONFIGURATION
# =====================================================
MODEL_OPTIONS = {
    "Llama 3.3 70B (Recommended - Reasoning)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Fast - Instant)": "llama-3.1-8b-instant",
    "Qwen 2.5 32B (Logic & Math)": "qwen/qwen3-32b",
    "MoonshotAI Kimi K2 (Long Context Specialist)": "moonshotai/kimi-k2-instruct-0905" 
}
GUARDRAIL_MODEL = "llama-guard-3-8b" 

# =====================================================
# 3. INITIALIZATION
# =====================================================
groq_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("❌ GROQ_API_KEY tidak ditemukan di secrets.toml")
    st.stop()

client = Groq(api_key=groq_key)

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    documents = []
    
    # Load CSV
    try:
        df = pd.read_csv("datasets/dokumen_produk.csv")
        if "product_url" in df.columns:
            df = df[df["product_url"].notna()]
        for _, row in df.iterrows():
            text_content = str(row.iloc[1])
            price_info = ""
            if 'price' in df.columns and pd.notna(row['price']):
                price_info = f" [HARGA: {row['price']}]"
            elif 'harga' in df.columns and pd.notna(row['harga']):
                price_info = f" [HARGA: {row['harga']}]"
            doc_final = f"[SUMBER: DATABASE PRODUK]{price_info} {text_content}"
            documents.append(doc_final)
    except Exception as e:
        st.error(f"Gagal memuat CSV: {e}")

    # Load PDF
    pdf_path = "datasets/jurnal_chunk.pdf"
    if os.path.exists(pdf_path):
        try:
            reader = PyPDF2.PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            chunk_size = 700
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i+chunk_size].replace("\n", " ")
                documents.append(f"[SUMBER: JURNAL DERMATOLOGI] {chunk}")
        except Exception as e:
            st.error(f"Gagal memuat PDF: {e}")

    if documents:
        embeddings = embedder.encode(documents, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return embedder, index, documents
    else:
        return None, None, []

embedder, faiss_index, all_documents = load_resources()

# =====================================================
# 4. GUARDRAILS
# =====================================================
def is_skincare_topic(text):
    keywords = [
        "kulit", "wajah", "jerawat", "kusam", "minyak", "kering", 
        "sensitif", "breakout", "skincare", "toner", "serum", 
        "sunscreen", "sabun", "bekas", "flek", "pori", "budget", "harga", "murah", "mahal"
    ]
    return any(k in text.lower() for k in keywords)

def check_safety_llamaguard(text):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model=GUARDRAIL_MODEL,
        )
        response = chat_completion.choices[0].message.content
        return "unsafe" not in response.lower()
    except Exception:
        return True 

# =====================================================
# 5. CORE LOGIC
# =====================================================
def retrieve_docs(query, k=5):
    if not faiss_index: return []
    qv = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    _, idxs = faiss_index.search(qv, k)
    return [all_documents[i] for i in idxs[0]]

def generate_recommendation(model_id, user_query, context):
    SYSTEM_PROMPT = """
    Anda adalah 'GlowAI', Asisten Dermatologi AI Profesional.
    INSTRUKSI UTAMA:
    1. Jawab HANYA berdasarkan KONTEN yang diberikan. Jangan halusinasi.
    2. Chain-of-Thought: Analisis kondisi kulit -> Cek kandungan -> Cek Harga/Budget -> Pilih Produk.
    
    ATURAN HARGA & BUDGET (STRICT):
    1. WAJIB MENAMPILKAN HARGA untuk setiap produk.
    2. FILTER BUDGET: Jika user menetapkan budget, HANYA rekomendasikan produk yang harganya masuk akal.
    3. PENANGANAN KEKURANGAN DATA: Jika jumlah produk kurang dari permintaan user, tuliskan disclaimer:
       "Mohon maaf, saya hanya dapat menemukan produk ini dari database yang sesuai dengan budget dan kriteria Anda."
    
    FORMAT OUTPUT:
    - **Analisis Kulit:** [Analisis singkat]
    - **Rekomendasi Rutinitas:**
      1. **[Nama Produk]**
         - **Harga:** [Sebutkan Harga]
         - *Kandungan Kunci:* ...
         - *Alasan Medis:* ...
         - [Link Produk]
    """
    USER_MESSAGE = f"""
    PERTANYAAN USER: {user_query}
    KONTEN REFERENSI:
    {context}
    """
    res = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE}
        ],
        temperature=0.3, 
        max_tokens=1500
    )
    return res.choices[0].message.content

# =====================================================
# 6. MAIN UI
# =====================================================
st.markdown('<div class="main-header"><h1>✨ GlowAI Skincare Expert</h1><p>Sistem Rekomendasi Berbasis Hybrid RAG & Guardrails</p></div>', unsafe_allow_html=True)

# Model selector moved to main page
st.subheader("⚙️ Pilih Model Generatif")
selected_model_name = st.selectbox("", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[selected_model_name]

user_input = st.text_area("💬 Ceritakan keluhan & budget Anda:", placeholder="Contoh: Saya cari toner untuk kulit berjerawat dan kusam. Budget saya di bawah 100 ribu.")

if st.button(" Analisis & Rekomendasi", type="primary"):
    if not user_input.strip():
        st.toast("⚠️ Mohon isi keluhan terlebih dahulu.")
    else:
        # --- GUARDRAILS ---
        with st.status("🛡️ Menjalankan Protokol Keamanan...", expanded=True) as status:
            if not is_skincare_topic(user_input):
                status.update(label="❌ Topik Tidak Relevan", state="error")
                st.error("Maaf, saya hanya dapat menjawab pertanyaan seputar skincare.")
                st.stop()
            if not check_safety_llamaguard(user_input):
                status.update(label="❌ Input Tidak Aman", state="error")
                st.error("Input terdeteksi melanggar kebijakan keamanan.")
                st.stop()
            status.update(label="✅ Input Valid & Aman", state="complete")

        # --- RETRIEVAL ---
        with st.spinner("📚 Mencari produk sesuai budget & jurnal relevan..."):
            retrieved_docs = retrieve_docs(user_input, k=15)
            if not retrieved_docs:
                st.warning("Maaf, tidak ditemukan data produk yang relevan.")
                st.stop()
            context_str = "\n\n".join(retrieved_docs)

        # --- GENERATION ---
        with st.spinner(f"🤖 Mengenerate jawaban dengan {selected_model_name}..."):
            try:
                response = generate_recommendation(model_id, user_input, context_str)
                st.markdown(f'<div class="result-card">{response}</div>', unsafe_allow_html=True)
                
                with st.expander("🔎 Lihat Sumber & Harga (Transparansi)"):
                    for doc in retrieved_docs:
                        st.caption(doc[:300] + "...") 
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan pada LLM: {e}")

st.markdown("---")
st.caption("© 2025 GlowAI Research Project. Dibuat untuk Tugas Akhir Teknik Informatika.")