import os
import re
import html
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import PyPDF2
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# =====================================================
# 1. PAGE CONFIG & STYLE
# =====================================================
st.set_page_config(
    page_title="GlowAI - Skincare Expert",
    page_icon="✨",
    layout="centered",
)

st.markdown(
    """
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
    color:#856404;
}
.info-card {
    background:#eef6ff;
    padding:15px;
    border-radius:10px;
    border:1px solid #cfe2ff;
    color:#084298;
}
small.muted { color:#666666; }
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# 2. FINAL MODEL CONFIGURATION
# =====================================================
FINAL_MODEL_NAME = "Llama 3.3 70B"
FINAL_MODEL_ID = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# =====================================================
# 3. INITIALIZATION
# =====================================================
groq_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("❌ GROQ_API_KEY tidak ditemukan di secrets.toml")
    st.stop()

client = Groq(api_key=groq_key)


def safe_float_price(value: object) -> Optional[float]:
    """
    Convert various price formats to float.
    Supports:
    - 50000
    - 50.000
    - Rp 50.000
    - Rp50,000
    Returns None if parsing fails.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    # keep digits only
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None

    try:
        return float(digits)
    except ValueError:
        return None


def format_rupiah(value: Optional[float]) -> str:
    if value is None:
        return "Harga tidak tersedia"
    return f"Rp{int(value):,}".replace(",", ".")


def extract_budget(query: str) -> Optional[float]:
    """
    Extract budget from Indonesian query.
    Examples:
    - di bawah 50rb
    - budget 100 ribu
    - under 150000
    - < 50000
    """
    q = query.lower().strip().replace(".", "")

    patterns = [
        r"(?:dibawah|di bawah|kurang dari|under|maksimal|max|budget)\s*rp?\s*(\d+)\s*(rb|ribu|k)?",
        r"<\s*rp?\s*(\d+)\s*(rb|ribu|k)?",
        r"budget\s*(?:di bawah)?\s*rp?\s*(\d+)\s*(rb|ribu|k)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            amount = float(match.group(1))
            unit = match.group(2)
            if unit in {"rb", "ribu", "k"}:
                amount *= 1000
            return amount

    return None


def is_skincare_topic(text: str) -> bool:
    """
    Rule-based topical guardrail.
    """
    keywords = [
        "kulit", "wajah", "jerawat", "kusam", "beruntusan", "komedo",
        "minyak", "berminyak", "kering", "sensitif", "iritasi", "pori",
        "bekas jerawat", "flek", "skin barrier", "skincare", "serum",
        "toner", "moisturizer", "pelembap", "facial wash", "cleanser",
        "face wash", "sunscreen", "exfoliating", "salicylic", "niacinamide",
        "budget", "harga", "murah", "mahal"
    ]
    lowered = text.lower()
    return any(k in lowered for k in keywords)


def detect_requested_category(query: str) -> Optional[str]:
    q = query.lower()
    category_map = {
        "serum": ["serum"],
        "toner": ["toner", "exfoliating toner"],
        "moisturizer": ["moisturizer", "pelembap", "pelembab"],
        "facial wash": ["facial wash", "face wash", "cleanser", "sabun wajah"],
    }
    for canonical, aliases in category_map.items():
        if any(alias in q for alias in aliases):
            return canonical
    return None


@st.cache_resource
def load_resources() -> Tuple[SentenceTransformer, faiss.IndexFlatIP, List[dict], pd.DataFrame]:
    """
    Load processed CSV + PDF knowledge base, then build FAISS index.

    Expected CSV: datasets/dokumen_produk.csv
    Recommended columns:
    - category
    - brand
    - product_name
    - price
    - product_url
    - serialized_text or document/text in second column
    """
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    documents: List[dict] = []

    csv_path = "datasets/dokumen_produk.csv"
    pdf_path = "datasets/jurnal_chunk.pdf"

    # -------- Load CSV product docs --------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan: {csv_path}")

    df = pd.read_csv(csv_path)

    # normalize columns
    df.columns = [str(c).strip() for c in df.columns]

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Flexible text source
        text_content = None
        for candidate in ["serialized_text", "document", "text", "dokumen", "content"]:
            if candidate in df.columns and pd.notna(row_dict.get(candidate)):
                text_content = str(row_dict[candidate]).strip()
                break

        if text_content is None:
            # fallback to 2nd column if available
            if len(df.columns) > 1 and pd.notna(row.iloc[1]):
                text_content = str(row.iloc[1]).strip()

        if not text_content:
            continue

        category = str(row_dict.get("category", "")).strip().lower()
        brand = str(row_dict.get("brand", "")).strip()
        product_name = str(row_dict.get("product_name", row_dict.get("name", ""))).strip()
        product_url = str(row_dict.get("product_url", "")).strip()
        raw_price = row_dict.get("price", row_dict.get("harga"))
        price_value = safe_float_price(raw_price)

        prefixed_text = (
            f"[SUMBER: DATABASE PRODUK] "
            f"[KATEGORI: {category or 'tidak diketahui'}] "
            f"[NAMA: {product_name or 'tidak diketahui'}] "
            f"[BRAND: {brand or 'tidak diketahui'}] "
            f"[HARGA: {format_rupiah(price_value)}] "
            f"{text_content}"
        )

        documents.append(
            {
                "source": "produk",
                "category": category,
                "product_name": product_name,
                "brand": brand,
                "price": price_value,
                "product_url": product_url,
                "text": prefixed_text,
            }
        )

    # -------- Load PDF dermatology docs --------
    if os.path.exists(pdf_path):
        reader = PyPDF2.PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            full_text += extracted + "\n"

        chunk_size = 700
        overlap = 50
        start = 0
        while start < len(full_text):
            chunk = full_text[start:start + chunk_size].replace("\n", " ").strip()
            if chunk:
                documents.append(
                    {
                        "source": "jurnal",
                        "category": "",
                        "product_name": "",
                        "brand": "",
                        "price": None,
                        "product_url": "",
                        "text": f"[SUMBER: JURNAL DERMATOLOGI] {chunk}",
                    }
                )
            start += max(1, chunk_size - overlap)

    if not documents:
        raise ValueError("Tidak ada dokumen yang berhasil dimuat.")

    corpus = [doc["text"] for doc in documents]
    embeddings = embedder.encode(corpus, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embedder, index, documents, df


try:
    embedder, faiss_index, all_documents, source_df = load_resources()
except Exception as e:
    st.error(f"Gagal memuat resource: {e}")
    st.stop()

# =====================================================
# 4. RULE-BASED GUARDRAILS
# =====================================================
def validate_input(query: str) -> Tuple[bool, str]:
    """
    Rule-based input guardrails aligned with Chapter IV:
    - domain restriction to skincare
    - basic harmful/non-topic blocking
    """
    q = query.strip()
    if not q:
        return False, "Mohon isi keluhan atau kebutuhan skincare terlebih dahulu."

    if len(q) < 8:
        return False, "Input terlalu singkat. Mohon jelaskan kondisi kulit atau kebutuhan skincare Anda."

    if not is_skincare_topic(q):
        return False, "Maaf, sistem hanya melayani pertanyaan seputar skincare dan perawatan kulit."

    blocked_patterns = [
        r"\bdiagnosa\b",
        r"\bresep obat\b",
        r"\bobat keras\b",
        r"\bpenyakit berat\b",
        r"\boperasi\b",
    ]
    for pattern in blocked_patterns:
        if re.search(pattern, q.lower()):
            return False, "Sistem ini berfungsi sebagai pendukung keputusan skincare dan tidak memberikan diagnosis medis profesional."

    return True, "Input valid."


def is_output_valid(response: str) -> bool:
    """
    Simple output validation / fallback trigger.
    """
    if not response or len(response.strip()) < 50:
        return False

    required_markers = [
        "Analisis Kulit",
        "Rekomendasi",
    ]
    if not any(marker.lower() in response.lower() for marker in required_markers):
        return False

    suspicious_patterns = [
        r"saya tidak tahu",
        r"di luar konteks",
        r"tidak ada informasi",
    ]
    suspicious_hits = sum(bool(re.search(p, response.lower())) for p in suspicious_patterns)

    # allow some uncertainty, but not too much
    if suspicious_hits >= 2:
        return False

    return True


def fallback_response(user_query: str, budget: Optional[float]) -> str:
    budget_note = ""
    if budget is not None:
        budget_note = f" dengan batas harga sekitar {format_rupiah(budget)}"

    return f"""
### Analisis Kulit
Sistem belum dapat menghasilkan tiga rekomendasi yang sepenuhnya sesuai untuk kebutuhan Anda{budget_note} berdasarkan basis pengetahuan yang tersedia.

### Rekomendasi
Mohon perjelas kondisi kulit Anda, misalnya jenis masalah utama, kategori produk yang diinginkan, atau batas budget yang lebih fleksibel. Sistem hanya memberikan rekomendasi berdasarkan data produk dan referensi dermatologi yang tersedia pada basis pengetahuan.
""".strip()

# =====================================================
# 5. RETRIEVAL + RE-RANKING
# =====================================================
def base_retrieve_docs(query: str, k: int = 30) -> List[dict]:
    qv = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    scores, idxs = faiss_index.search(qv, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        doc = dict(all_documents[idx])
        doc["base_score"] = float(score)
        results.append(doc)
    return results


def rerank_docs(
    query: str,
    docs: List[dict],
    budget: Optional[float],
    requested_category: Optional[str],
    final_k: int = 15,
) -> List[dict]:
    """
    Simple rule-based hybrid retrieval aligned with Chapter IV:
    - semantic similarity from FAISS
    - category preference
    - budget preference / boost
    """
    reranked = []

    for doc in docs:
        score = doc.get("base_score", 0.0)

        # boost category match for product docs
        if requested_category and doc["source"] == "produk":
            if requested_category in (doc.get("category") or ""):
                score += 0.10

        # boost budget-compatible docs
        price = doc.get("price")
        if budget is not None and doc["source"] == "produk" and price is not None:
            if price <= budget:
                score += 0.20
            else:
                score -= 0.15

        # give slight priority to journal chunks to keep medical grounding
        if doc["source"] == "jurnal":
            score += 0.03

        doc["hybrid_score"] = score
        reranked.append(doc)

    reranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return reranked[:final_k]


def retrieve_docs(query: str, final_k: int = 15) -> List[dict]:
    budget = extract_budget(query)
    requested_category = detect_requested_category(query)
    base_docs = base_retrieve_docs(query, k=30)
    final_docs = rerank_docs(
        query=query,
        docs=base_docs,
        budget=budget,
        requested_category=requested_category,
        final_k=final_k,
    )
    return final_docs


def build_context(docs: List[dict]) -> str:
    """
    Context builder for LLM.
    """
    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        context_blocks.append(f"[Dokumen {i}]\n{doc['text']}")
    return "\n\n".join(context_blocks)

# =====================================================
# 6. GENERATION
# =====================================================
def generate_recommendation(user_query: str, context: str) -> str:
    system_prompt = """
Anda adalah GlowAI, asisten rekomendasi skincare berbasis data.

ATURAN UTAMA:
1. Jawab HANYA berdasarkan konteks yang diberikan.
2. Jangan menambahkan produk, harga, kandungan, atau klaim yang tidak ada di konteks.
3. Fokuskan alasan rekomendasi pada masalah utama pengguna.
4. Jika pengguna menyebut budget, prioritaskan hanya produk yang sesuai budget.
5. Jika jumlah produk yang sesuai kurang dari 3, tulis catatan:
   "Rekomendasi disesuaikan dengan data yang tersedia pada basis pengetahuan sistem."
6. Jangan memberi diagnosis medis profesional.
7. Berikan jawaban yang ringkas, jelas, dan terstruktur.

FORMAT OUTPUT WAJIB:
### Analisis Kulit
[analisis singkat kondisi kulit pengguna]

### Rekomendasi
1. **Nama Produk**
   - **Harga:** ...
   - **Kandungan Kunci:** ...
   - **Alasan Medis:** ...
   - **Link Produk:** ...

2. **Nama Produk**
   - **Harga:** ...
   - **Kandungan Kunci:** ...
   - **Alasan Medis:** ...
   - **Link Produk:** ...

3. **Nama Produk**
   - **Harga:** ...
   - **Kandungan Kunci:** ...
   - **Alasan Medis:** ...
   - **Link Produk:** ...

### Catatan
[jika diperlukan]
""".strip()

    user_message = f"""
PERTANYAAN PENGGUNA:
{user_query}

KONTEKS:
{context}
""".strip()

    res = client.chat.completions.create(
        model=FINAL_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    return res.choices[0].message.content.strip()

# =====================================================
# 7. MAIN UI
# =====================================================
st.markdown(
    f"""
<div class="main-header">
    <h1>✨ GlowAI Skincare Expert</h1>
    <p>Sistem Rekomendasi Berbasis RAG, FAISS, dan {html.escape(FINAL_MODEL_NAME)}</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="info-card">
<b>Petunjuk:</b> Jelaskan keluhan kulit Anda dan, jika perlu, sertakan kategori produk atau budget.
Contoh: <i>"Rekomendasikan 3 serum untuk bekas jerawat dengan budget di bawah 50 ribu."</i>
</div>
""",
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "💬 Ceritakan keluhan atau kebutuhan skincare Anda:",
    placeholder=(
        "Contoh: Rekomendasikan 3 produk moisturizer untuk kulit kering yang terasa "
        "ketarik setelah mencuci muka."
    ),
    height=130,
)

if st.button("Analisis & Rekomendasi", type="primary"):
    is_valid, validation_msg = validate_input(user_input)
    if not is_valid:
        st.error(validation_msg)
        st.stop()

    budget = extract_budget(user_input)
    category = detect_requested_category(user_input)

    with st.status("🛡️ Menjalankan validasi input...", expanded=False) as status:
        status.write("Topik terdeteksi sesuai domain skincare.")
        if budget is not None:
            status.write(f"Budget terdeteksi: {format_rupiah(budget)}")
        if category is not None:
            status.write(f"Kategori produk terdeteksi: {category}")
        status.update(label="✅ Validasi input selesai", state="complete")

    with st.spinner("📚 Melakukan semantic retrieval dan re-ranking..."):
        retrieved_docs = retrieve_docs(user_input, final_k=15)

    if not retrieved_docs:
        st.warning("Tidak ditemukan dokumen yang relevan pada basis pengetahuan.")
        st.markdown(
            f'<div class="warning-card">{fallback_response(user_input, budget)}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    context_str = build_context(retrieved_docs)

    with st.spinner(f"🤖 Menghasilkan rekomendasi dengan {FINAL_MODEL_NAME}..."):
        try:
            response = generate_recommendation(user_input, context_str)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat generate jawaban: {e}")
            st.stop()

    # Output validation + fallback
    if not is_output_valid(response):
        response = fallback_response(user_input, budget)

    st.markdown(f'<div class="result-card">{response}</div>', unsafe_allow_html=True)

    with st.expander("🔎 Transparansi Dokumen Retrieval"):
        for i, doc in enumerate(retrieved_docs, start=1):
            meta = (
                f"Skor: {doc.get('hybrid_score', 0):.3f} | "
                f"Sumber: {doc.get('source', '-')}"
            )
            if doc.get("price") is not None:
                meta += f" | Harga: {format_rupiah(doc['price'])}"
            if doc.get("category"):
                meta += f" | Kategori: {doc['category']}"
            st.markdown(f"**Dokumen {i}** — {meta}")
            st.caption(doc["text"][:400] + ("..." if len(doc["text"]) > 400 else ""))

st.markdown("---")
st.caption("© 2025 GlowAI Research Project. Prototipe sistem rekomendasi skincare berbasis RAG.")