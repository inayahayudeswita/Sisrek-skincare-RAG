import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# =====================================================
# 1. PAGE CONFIG
# =====================================================
st.set_page_config(page_title="GlowAI", page_icon="G", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #faf9f7;
    }

    /* ── Layout width ── */
    .block-container {
        max-width: 960px !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        padding-top: 4rem !important;
    }

    .main-wordmark {
        font-family: 'Cormorant Garamond', serif;
        font-size: 5.5rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: #1a1a1a;
        text-align: center;
        margin-top: 2.5rem;
        margin-bottom: 0.4rem;
    }

    .main-wordmark span {
        color: #b5936b;
    }

    .subtitle {
        text-align: center;
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        font-weight: 300;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #8c8070;
        margin-bottom: 3.5rem;
    }

    .divider {
        border: none;
        border-top: 1px solid #e4ddd5;
        margin: 2rem 0;
    }

    .info-box {
        background: #ffffff;
        padding: 36px 44px;
        border-radius: 4px;
        border: 1px solid #e4ddd5;
        margin-bottom: 2rem;
        font-size: 1.1rem;
        color: #3d3530;
        line-height: 1.8;
    }

    .info-box .info-title {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #1a1a1a;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }

    .info-box ul {
        margin: 0.75rem 0 0 0;
        padding-left: 1.3rem;
    }

    .info-box li {
        margin-bottom: 0.5rem;
        color: #5a5048;
        font-size: 1.05rem;
    }

    .category-chips {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 1rem 0 0.25rem 0;
    }

    .chip {
        background: #f3ede7;
        border: 1px solid #d9cec5;
        border-radius: 2px;
        padding: 7px 18px;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b5d52;
    }

    .result-box {
        background: #ffffff;
        padding: 40px 48px;
        border-radius: 4px;
        border: 1px solid #d9cec5;
        border-left: 3px solid #b5936b;
        margin-top: 2rem;
        font-size: 1.1rem;
        color: #2e2825;
        line-height: 2.0;
    }

    .warning-box {
        background: #fffbf5;
        padding: 20px 24px;
        border-radius: 4px;
        border: 1px solid #e8d5b0;
        margin-top: 1.25rem;
        font-size: 0.95rem;
        color: #6b5228;
    }

    .meta-strip {
        font-size: 0.8rem;
        color: #9d9088;
        letter-spacing: 0.04em;
        margin-top: 1rem;
    }

    div[data-testid="stTextArea"] textarea {
        border: 1px solid #d9cec5 !important;
        border-radius: 4px !important;
        background: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.1rem !important;
        color: #2e2825 !important;
        padding: 20px 24px !important;
        line-height: 1.7 !important;
        box-shadow: none !important;
        min-height: 160px !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #b5936b !important;
        box-shadow: 0 0 0 3px rgba(181, 147, 107, 0.12) !important;
    }

    div[data-testid="stTextArea"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.13em !important;
        text-transform: uppercase !important;
        color: #7a6d64 !important;
        margin-bottom: 8px !important;
    }

    .stButton > button {
        background-color: #1a1a1a !important;
        color: #faf9f7 !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        padding: 16px 52px !important;
        transition: background 0.2s ease !important;
        margin-top: 0.5rem !important;
    }

    .stButton > button:hover {
        background-color: #b5936b !important;
    }

    .stSpinner > div {
        border-top-color: #b5936b !important;
    }

    .stExpander {
        border: 1px solid #e4ddd5 !important;
        border-radius: 4px !important;
        background: #ffffff !important;
        margin-top: 1.5rem !important;
    }

    .stExpander summary, .stExpander [data-testid="stExpanderToggleIcon"] {
        font-size: 0.85rem !important;
        letter-spacing: 0.05em !important;
        color: #7a6d64 !important;
    }

    footer { visibility: hidden; }

    .footer-note {
        text-align: center;
        font-size: 0.75rem;
        color: #b0a89e;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 4rem;
        padding-bottom: 2.5rem;
    }

    /* ── MOBILE RESPONSIVE ── */
    @media (max-width: 768px) {
        .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1.5rem !important;
        }

        .main-wordmark {
            font-size: 3rem !important;
            margin-top: 1rem !important;
            margin-bottom: 0.2rem !important;
        }

        .subtitle {
            font-size: 0.7rem !important;
            letter-spacing: 0.14em !important;
            margin-bottom: 1.5rem !important;
        }

        .info-box {
            padding: 20px 18px !important;
            font-size: 0.9rem !important;
            margin-bottom: 1.25rem !important;
        }

        .info-box .info-title {
            font-size: 0.85rem !important;
        }

        .info-box li {
            font-size: 0.88rem !important;
            margin-bottom: 0.4rem !important;
        }

        .chip {
            font-size: 0.7rem !important;
            padding: 5px 10px !important;
        }

        .result-box {
            padding: 20px 18px !important;
            font-size: 0.92rem !important;
            line-height: 1.75 !important;
            margin-top: 1.25rem !important;
        }

        .meta-strip {
            font-size: 0.7rem !important;
            line-height: 1.6 !important;
        }

        div[data-testid="stTextArea"] textarea {
            font-size: 0.95rem !important;
            padding: 12px 14px !important;
        }

        div[data-testid="stTextArea"] label {
            font-size: 0.75rem !important;
        }

        .stButton > button {
            width: 100% !important;
            font-size: 0.8rem !important;
            padding: 13px 20px !important;
        }

        .footer-note {
            font-size: 0.65rem !important;
            margin-top: 2rem !important;
        }
    }

    @media (max-width: 480px) {
        .main-wordmark {
            font-size: 2.4rem !important;
        }

        .subtitle {
            font-size: 0.62rem !important;
            letter-spacing: 0.1em !important;
        }

        .info-box {
            padding: 16px 14px !important;
        }

        .result-box {
            padding: 16px 14px !important;
            font-size: 0.88rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">',
    unsafe_allow_html=True,
)
st.markdown("<div class='main-wordmark'>Glow<span>AI</span></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Sistem Rekomendasi Skincare &mdash; RAG &middot; FAISS &middot; LLM</div>",
    unsafe_allow_html=True,
)

# =====================================================
# 2. CONFIG
# =====================================================
PRIMARY_MODEL_ID = "llama-3.3-70b-versatile"
BACKUP_MODEL_ID = "openai/gpt-oss-120b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Google Drive file IDs
GDRIVE_PRODUK_ID  = "1uzt2pxhB4WBmQySfNAh3V40YIvXynmLs"
GDRIVE_JURNAL_ID  = "1Sb2lgFLrWaAIyBx6BP3t1GbtIBa5lHXM"

CSV_PATH          = "datasets/dokumen_produk.csv"
JOURNAL_CSV_PATH  = "datasets/jurnal_chunk.csv"

INITIAL_RETRIEVAL = 50
TOPK_CONTEXT = 5
MIN_SIMILARITY_SCORE = 0.25

MAX_INPUT_CHARS = 500
DEFAULT_RECOMMENDATION_COUNT = 3
MAX_RECOMMENDATION_COUNT = 5
MAX_CONTEXT_CHARS_PER_DOC = 1200
MAX_OUTPUT_TOKENS = 1200

ALLOWED_CATEGORIES = ["facewash", "toner", "serum", "moisturizer"]

# =====================================================
# 2b. DATASET DOWNLOAD FROM GOOGLE DRIVE
# =====================================================
def _gdrive_direct(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


@st.cache_resource(show_spinner=False)
def ensure_datasets() -> None:
    """Download CSVs from Google Drive if not already present locally."""
    import urllib.request

    pairs = [
        (GDRIVE_PRODUK_ID,  CSV_PATH),
        (GDRIVE_JURNAL_ID,  JOURNAL_CSV_PATH),
    ]

    for file_id, local_path in pairs:
        path = Path(local_path)
        if path.exists():
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        url = _gdrive_direct(file_id)

        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            st.error(
                f"Gagal mengunduh dataset dari Google Drive: {local_path}\n"
                f"Error: {e}\n\n"
                "Pastikan file Google Drive sudah diset 'Anyone with the link can view'."
            )
            st.stop()


with st.spinner("Memuat dataset..."):
    ensure_datasets()

# =====================================================
# 3. CLIENT
# =====================================================
groq_key = st.secrets.get("GROQ_API_KEY", "")
if not groq_key:
    st.error("GROQ_API_KEY tidak ditemukan di secrets.toml")
    st.stop()

openrouter_key = st.secrets.get("OPENROUTER_API_KEY", "")

groq_client = Groq(api_key=groq_key)

openrouter_client = (
    OpenAI(
        api_key=openrouter_key,
        base_url=OPENROUTER_BASE_URL,
    )
    if openrouter_key
    else None
)

# =====================================================
# 4. BASIC HELPERS
# =====================================================
def format_rp(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "Harga tidak tersedia"
    return f"Rp{int(x):,}".replace(",", ".")


def parse_price(val: Any) -> Optional[float]:
    if val is None or pd.isna(val):
        return None

    raw = str(val).lower().strip()

    if raw in ["", "-", "nan", "none", "tidak tersedia", "harga tidak tersedia"]:
        return None

    digits = re.sub(r"[^\d]", "", raw)

    if not digits:
        return None

    price = float(digits)

    if price <= 0:
        return None

    return price


def extract_field(text: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}\s*:\s*(.*)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def normalize_category(category: str) -> str:
    c = str(category).lower().strip()

    if "serum" in c:
        return "serum"
    if "toner" in c:
        return "toner"
    if "moisturizer" in c or "pelembap" in c or "pelembab" in c:
        return "moisturizer"
    if "facewash" in c or "face wash" in c or "cleanser" in c or "sabun wajah" in c:
        return "facewash"

    return c


def extract_budget(text: str) -> Optional[float]:
    t = text.lower().replace(".", "").replace(",", "").strip()

    patterns = [
        r"(?:di bawah|dibawah|kurang dari|maksimal|max|under|<)\s*(?:rp\s*)?(\d+)\s*(rb|ribu|k|000)?",
        r"budget\s*(?:di bawah|dibawah|kurang dari|maksimal|max|under|<)?\s*(?:rp\s*)?(\d+)\s*(rb|ribu|k|000)?",
        r"(?:harga|budget)\s*(?:maksimal|max)?\s*(?:rp\s*)?(\d+)\s*(rb|ribu|k|000)?",
    ]

    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            amount = float(m.group(1))
            unit = m.group(2)

            if unit in {"rb", "ribu", "k", "000"}:
                amount *= 1000

            if amount < 1000:
                amount *= 1000

            return amount

    return None


def extract_requested_count(text: str) -> int:
    t = text.lower()

    patterns = [
        r"(?:rekomendasikan|berikan|kasih|tampilkan|carikan)\s+(\d+)",
        r"(\d+)\s+(?:produk|serum|toner|moisturizer|pelembap|pelembab|cleanser|facial wash|face wash)",
    ]

    for pattern in patterns:
        m = re.search(pattern, t)
        if m:
            try:
                n = int(m.group(1))
                return max(1, min(n, MAX_RECOMMENDATION_COUNT))
            except ValueError:
                pass

    return DEFAULT_RECOMMENDATION_COUNT


def detect_category(text: str) -> Optional[str]:
    t = text.lower()

    mapping = {
        "serum": ["serum"],
        "toner": ["toner", "exfoliating toner"],
        "moisturizer": ["moisturizer", "pelembap", "pelembab"],
        "facewash": ["facewash", "facial wash", "face wash", "cleanser", "sabun wajah"],
    }

    for cat, aliases in mapping.items():
        if any(alias in t for alias in aliases):
            return cat

    return None


def is_skincare_topic(text: str) -> bool:
    keywords = [
        "kulit", "wajah", "skincare", "jerawat", "beruntusan", "bruntusan",
        "komedo", "kusam", "kering", "berminyak", "sensitif", "iritasi",
        "pori", "bekas jerawat", "flek", "noda hitam", "acne",
        "oily", "dry skin", "sensitive skin", "mengencangkan", "kendur",
        "anti aging", "anti-aging", "kerutan", "garis halus",
        "serum", "toner", "moisturizer", "pelembap", "pelembab",
        "cleanser", "facial wash", "facewash", "sabun wajah",
        "budget", "harga", "murah", "normal", "t-zone", "kombinasi",
    ]

    t = text.lower()
    return any(k in t for k in keywords)


def validate_input(text: str) -> Tuple[bool, str]:
    cleaned = text.strip()

    if not cleaned:
        return False, "Mohon isi pertanyaan terlebih dahulu."

    if len(cleaned) < 8:
        return False, "Input terlalu singkat."

    if len(cleaned) > MAX_INPUT_CHARS:
        return False, f"Input terlalu panjang. Maksimal {MAX_INPUT_CHARS} karakter."

    if not is_skincare_topic(cleaned):
        return False, "GlowAI hanya melayani pertanyaan seputar skincare."

    medical_terms = r"\b(diagnosa|diagnosis|obat keras|resep|penyakit serius|psoriasis|eksim|infeksi|nanah parah)\b"
    if re.search(medical_terms, cleaned.lower()):
        return False, "GlowAI tidak memberikan diagnosis medis profesional. Silakan konsultasikan kondisi tersebut ke dokter."

    return True, ""


def valid_output(text: str) -> bool:
    if not text or len(text.strip()) < 80:
        return False

    lowered = text.lower()
    return ("analisis kulit" in lowered) and ("rekomendasi" in lowered)


def answer_looks_incomplete(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()

    if not stripped:
        return True

    incomplete_endings = (
        "link",
        "link produk",
        "alasan medis",
        "kandungan kunci",
        "harga",
        "dengan",
        "dan",
        "serta",
        "karena",
        "untuk",
        "http",
        "https://reviews.f",
    )

    if any(lowered.endswith(x) for x in incomplete_endings):
        return True

    return "### catatan" not in lowered and "catatan" not in lowered


def answer_violates_budget(answer: str, budget: Optional[float]) -> bool:
    if budget is None:
        return False

    price_patterns = [
        r"rp\.?\s*(\d{1,3}(?:[.,]\d{3})+|\d+)",
        r"harga\s*:\s*(\d{1,3}(?:[.,]\d{3})+|\d+)",
    ]

    for pattern in price_patterns:
        for m in re.finditer(pattern, answer.lower()):
            price = parse_price(m.group(1))
            if price is not None and price > budget:
                return True

    return False


def shorten_text(text: str, max_chars: int = MAX_CONTEXT_CHARS_PER_DOC) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def count_product_docs(docs: List[Dict[str, Any]]) -> int:
    return sum(1 for d in docs if d.get("source") == "produk")


def count_journal_docs(docs: List[Dict[str, Any]]) -> int:
    return sum(1 for d in docs if d.get("source") == "jurnal")


def product_matches_constraints(
    doc: Dict[str, Any],
    category: Optional[str],
    budget: Optional[float],
) -> bool:
    if doc.get("source") != "produk":
        return False

    if category and doc.get("category") != category:
        return False

    if budget is not None:
        price = doc.get("price")
        if price is None or price > budget:
            return False

    return True


def select_context_docs(
    product_results: List[Dict[str, Any]],
    journal_results: List[Dict[str, Any]],
    k: int = TOPK_CONTEXT,
) -> List[Dict[str, Any]]:

    journal_slots = 1 if journal_results and product_results else 0
    product_slots = max(0, k - journal_slots)

    selected = product_results[:product_slots]

    if journal_slots:
        selected.append(journal_results[0])

    return selected[:k]


# =====================================================
# 5. CLINICAL HELPERS
# =====================================================
def detect_skin_concerns(text: str) -> List[str]:
    t = text.lower()
    concerns = []

    if any(k in t for k in ["jerawat meradang", "jerawat radang", "nyeri", "sakit", "bengkak", "meradang"]):
        concerns.append("inflamed_acne")

    if any(k in t for k in ["jerawat", "acne", "komedo"]):
        concerns.append("acne")

    if any(k in t for k in ["bruntusan", "beruntusan", "closed comedone", "komedo tertutup"]):
        concerns.append("comedonal_acne")

    if any(k in t for k in ["bekas jerawat", "noda hitam", "flek", "hiperpigmentasi", "kusam", "hitam"]):
        concerns.append("hyperpigmentation")

    if any(k in t for k in ["kering", "ketarik", "dehidrasi", "mengelupas"]):
        concerns.append("dry_skin")

    if any(k in t for k in ["sensitif", "iritasi", "mudah merah", "perih", "reaktif"]):
        concerns.append("sensitive_skin")

    if any(k in t for k in ["pori", "pori-pori"]):
        concerns.append("large_pores")

    if any(k in t for k in ["kulit normal", "normal skin", "menjaga kesehatan", "maintenance"]):
        concerns.append("normal_skin")

    if any(k in t for k in ["kombinasi", "t-zone", "t zone", "pipi kering"]):
        concerns.append("combination_skin")

    if any(k in t for k in ["berminyak", "oily", "minyak berlebih"]):
        concerns.append("oily_skin")

    return list(set(concerns))


def clinical_score_boost(doc_text: str, concerns: List[str]) -> float:
    t = doc_text.lower()
    boost = 0.0

    beneficial = {
        "inflamed_acne": [
            "benzoyl peroxide", "salicylic acid", "bha", "azelaic acid",
            "niacinamide", "centella", "cica", "sulfur", "zinc",
        ],
        "acne": [
            "salicylic acid", "bha", "niacinamide", "zinc", "tea tree",
            "azelaic acid", "centella", "cica",
        ],
        "comedonal_acne": [
            "salicylic acid", "bha", "retinol", "glycolic acid", "lactic acid",
        ],
        "hyperpigmentation": [
            "vitamin c", "niacinamide", "alpha arbutin", "arbutin",
            "azelaic acid", "tranexamic acid", "licorice", "retinol",
        ],
        "dry_skin": [
            "hyaluronic acid", "glycerin", "ceramide", "panthenol",
            "squalane", "aloe", "centella", "cica",
        ],
        "sensitive_skin": [
            "centella", "cica", "panthenol", "ceramide", "aloe",
            "hyaluronic acid", "oat", "allantoin",
        ],
        "large_pores": [
            "salicylic acid", "bha", "niacinamide", "zinc", "retinol",
        ],
        "combination_skin": [
            "hyaluronic acid", "glycerin", "niacinamide", "ceramide",
            "gel", "lightweight", "ringan",
        ],
        "normal_skin": [
            "hyaluronic acid", "glycerin", "ceramide", "centella",
            "panthenol", "moisturizing", "hydrating",
        ],
        "oily_skin": [
            "niacinamide", "salicylic acid", "bha", "zinc", "oil control",
            "sebum",
        ],
    }

    risky = {
        "sensitive_skin": [
            "retinol", "aha", "bha", "glycolic acid", "lactic acid",
            "fragrance", "parfum", "alcohol denat",
        ],
        "dry_skin": [
            "alcohol denat", "high alcohol", "strong exfoliating",
        ],
        "normal_skin": [
            "retinol", "aha", "bha", "salicylic acid", "glycolic acid",
            "lactic acid", "exfoliating",
        ],
        "inflamed_acne": [
            "bengkoang", "whitening only", "brightening only",
        ],
    }

    for concern in concerns:
        for ingredient in beneficial.get(concern, []):
            if ingredient in t:
                boost += 0.08

        for ingredient in risky.get(concern, []):
            if ingredient in t:
                boost -= 0.10

    if any(c in concerns for c in ["inflamed_acne", "acne", "comedonal_acne"]) and "bengkoang" in t:
        boost -= 0.12

    return boost


def generate_safety_notes(concerns: List[str], category: Optional[str]) -> str:
    notes = []

    if "hyperpigmentation" in concerns:
        notes.append("- Untuk bekas jerawat/noda hitam, gunakan sunscreen pada pagi/siang hari agar hasil perawatan lebih optimal.")

    if "sensitive_skin" in concerns:
        notes.append("- Untuk kulit sensitif, pilih produk yang minim iritan dan lakukan patch test sebelum pemakaian rutin.")

    if "dry_skin" in concerns:
        notes.append("- Untuk kulit kering yang terasa ketarik, fokus pada hidrasi dan perbaikan skin barrier.")

    if "inflamed_acne" in concerns:
        notes.append("- Jika jerawat meradang terasa nyeri, luas, bernanah, atau tidak membaik, sebaiknya konsultasi ke dokter kulit.")

    if "comedonal_acne" in concerns:
        notes.append("- Produk eksfoliasi sebaiknya digunakan bertahap, misalnya 2-3 kali per minggu terlebih dahulu.")

    if "large_pores" in concerns:
        notes.append("- Pori-pori tidak dapat dikecilkan permanen, tetapi tampilannya dapat dibantu dengan kontrol minyak dan perawatan pori.")

    if "normal_skin" in concerns:
        notes.append("- Untuk kulit normal, hindari penggunaan bahan aktif terlalu banyak agar tidak terjadi over-treatment.")

    if category == "toner":
        notes.append("- Hindari penggunaan toner eksfoliasi bersamaan dengan retinol atau exfoliant lain pada malam yang sama.")

    return "\n".join(notes[:2])


def fallback_response(
    jumlah_produk: int,
    category: Optional[str] = None,
    budget: Optional[float] = None,
    concerns: Optional[List[str]] = None,
) -> str:
    category_text = f" kategori **{category}**" if category else ""
    budget_text = f" dengan budget **di bawah {format_rp(budget)}**" if budget is not None else ""
    safety = generate_safety_notes(concerns or [], category)

    return f"""### Analisis Kulit
Sistem belum menemukan rekomendasi{category_text}{budget_text} yang sepenuhnya sesuai berdasarkan basis pengetahuan yang tersedia.

### Rekomendasi
Tidak ditemukan produk yang memenuhi seluruh batasan pengguna pada data yang tersedia.

### Catatan
Silakan perluas budget, ubah kategori produk, atau jelaskan kondisi kulit dengan lebih spesifik. Rekomendasi disesuaikan dengan data yang tersedia pada basis pengetahuan sistem.
{safety}
"""


# =====================================================
# 6. LOAD RESOURCES
# =====================================================
@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(EMBED_MODEL)

    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    docs: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        text = str(row.get("dokumen", "")).strip()
        if not text:
            continue
        if "discontinued" in text.lower():
            continue

        name = extract_field(text, "Nama Produk")
        brand = extract_field(text, "Brand")
        category_raw = extract_field(text, "Kategori")
        category = normalize_category(category_raw)

        raw_price = extract_field(text, "Harga")
        url = (
            extract_field(text, "Link Produk")
            or extract_field(text, "Link")
            or extract_field(text, "URL")
        )

        price = parse_price(raw_price)

        if category and category not in ALLOWED_CATEGORIES:
            continue

        full = (
            f"[SUMBER: DATABASE PRODUK] "
            f"[KATEGORI: {category}] "
            f"[NAMA: {name}] "
            f"[BRAND: {brand}] "
            f"[HARGA: {raw_price if raw_price else 'Harga tidak tersedia'}] "
            f"[LINK: {url if url else 'Link tidak tersedia'}] "
            f"{text}"
        )

        docs.append({
            "source": "produk",
            "category": category,
            "product_name": name,
            "brand": brand,
            "price": price,
            "raw_price": raw_price,
            "url": url,
            "text": full,
        })

    if os.path.exists(JOURNAL_CSV_PATH):
        journal_df = pd.read_csv(JOURNAL_CSV_PATH)
        journal_df.columns = [str(c).strip() for c in journal_df.columns]

        for _, row in journal_df.iterrows():
            topic = str(row.get("topic", "general_dermatology")).strip()
            source_file = str(row.get("source_file", "unknown_source")).strip()
            page = str(row.get("page", "")).strip()
            chunk_id = str(row.get("new_chunk_id", row.get("chunk_id", ""))).strip()
            text = str(row.get("text", "")).strip()

            if not text:
                continue

            docs.append({
                "source": "jurnal",
                "category": "",
                "product_name": "",
                "brand": "",
                "price": None,
                "raw_price": "",
                "url": "",
                "topic": topic,
                "source_file": source_file,
                "page": page,
                "chunk_id": chunk_id,
                "text": (
                    f"[SUMBER: JURNAL DERMATOLOGI] "
                    f"[TOPIK: {topic}] "
                    f"[FILE: {source_file}] "
                    f"[HALAMAN: {page}] "
                    f"[CHUNK: {chunk_id}] "
                    f"{text}"
                ),
            })

    if not docs:
        raise ValueError("Tidak ada dokumen yang berhasil dimuat.")

    corpus = [d["text"] for d in docs]
    emb = embedder.encode(corpus, convert_to_numpy=True)
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    journal_docs = [d for d in docs if d.get("source") == "jurnal"]
    journal_index = None

    if journal_docs:
        journal_corpus = [d["text"] for d in journal_docs]
        journal_emb = embedder.encode(journal_corpus, convert_to_numpy=True)
        faiss.normalize_L2(journal_emb)

        journal_index = faiss.IndexFlatIP(journal_emb.shape[1])
        journal_index.add(journal_emb)

    return embedder, index, docs, journal_index, journal_docs


try:
    embedder, index, all_docs, journal_index, journal_docs = load_resources()
except Exception as e:
    st.error(f"Gagal memuat resource: {e}")
    st.stop()

st.markdown(
    f"<div class='meta-strip' style='text-align:center; margin-bottom: 2rem;'>"
    f"Total dokumen: {len(all_docs)} &nbsp;&middot;&nbsp; "
    f"Produk: {sum(1 for d in all_docs if d.get('source') == 'produk')} &nbsp;&middot;&nbsp; "
    f"Jurnal: {sum(1 for d in all_docs if d.get('source') == 'jurnal')}"
    f"</div>",
    unsafe_allow_html=True,
)


# =====================================================
# 7. RETRIEVAL
# =====================================================
def retrieve(query: str, k: int = TOPK_CONTEXT) -> List[Dict[str, Any]]:
    qv = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)

    scores, ids = index.search(qv, INITIAL_RETRIEVAL)

    budget = extract_budget(query)
    category = detect_category(query)
    concerns = detect_skin_concerns(query)

    product_results = []
    journal_results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue

        doc = dict(all_docs[idx])
        base_score = float(score)
        final_score = base_score

        if doc["source"] == "produk":
            if base_score < MIN_SIMILARITY_SCORE:
                continue

            if not product_matches_constraints(doc, category, budget):
                continue

            if budget is not None:
                final_score += 0.20

            final_score += clinical_score_boost(doc.get("text", ""), concerns)

            doc["base_score"] = base_score
            doc["score"] = final_score
            product_results.append(doc)

        elif doc["source"] == "jurnal":
            if base_score < 0.10:
                continue

            doc["base_score"] = base_score
            doc["score"] = final_score + 0.03
            journal_results.append(doc)

    if journal_index is not None and journal_docs:
        journal_scores, journal_ids = journal_index.search(qv, min(5, len(journal_docs)))
        existing_chunks = {d.get("chunk_id") for d in journal_results}

        for score, idx in zip(journal_scores[0], journal_ids[0]):
            if idx < 0:
                continue

            doc = dict(journal_docs[idx])
            if doc.get("chunk_id") in existing_chunks:
                continue

            doc["base_score"] = float(score)
            doc["score"] = float(score) + 0.03
            journal_results.append(doc)

    product_results.sort(key=lambda x: x["score"], reverse=True)
    journal_results.sort(key=lambda x: x["score"], reverse=True)

    return select_context_docs(product_results, journal_results, k)


def build_context(docs: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, d in enumerate(docs, 1):
        context_parts.append(
            f"[Dokumen {i}]\n{shorten_text(d['text'])}"
        )

    return "\n\n".join(context_parts)


# =====================================================
# 8. GENERATION
# =====================================================
def build_prompt(
    jumlah_produk: int,
    budget: Optional[float] = None,
    category: Optional[str] = None,
    concerns: Optional[List[str]] = None,
) -> str:
    concerns = concerns or []

    budget_rule = ""
    if budget is not None:
        budget_rule = f"""
ATURAN BUDGET WAJIB:
- Pengguna menyebut budget maksimal {format_rp(budget)}.
- JANGAN menampilkan produk dengan harga di atas {format_rp(budget)} dalam kondisi apa pun.
- JANGAN menampilkan produk dengan harga tidak tersedia jika pengguna menyebut budget.
- Jika tidak ada produk yang sesuai budget dalam konteks, tulis bahwa produk sesuai budget tidak ditemukan.
""".strip()

    category_rule = ""
    if category is not None:
        category_rule = f"""
ATURAN KATEGORI WAJIB:
- Pengguna meminta kategori {category}.
- JANGAN menampilkan produk dari kategori lain.
""".strip()

    clinical_rules = f"""
ATURAN DERMATOLOGI:
- Kondisi terdeteksi: {", ".join(concerns) if concerns else "tidak spesifik"}.
- Untuk jerawat meradang/nyeri: prioritaskan kandungan anti-inflamasi atau acne treatment seperti salicylic acid, benzoyl peroxide, azelaic acid, sulfur, zinc, niacinamide, centella/cica jika tersedia. Jangan hanya memilih produk pencerah.
- Untuk bruntusan/komedo: prioritaskan BHA/salicylic acid. Jika ada AHA/BHA, wajib beri catatan pemakaian bertahap.
- Untuk bekas jerawat/kusam/PIH: prioritaskan vitamin C, niacinamide, alpha arbutin, azelaic acid, tranexamic acid, atau retinol jika tersedia. Wajib sebut sunscreen pada catatan.
- Untuk kulit kering/ketarik: prioritaskan hyaluronic acid, glycerin, ceramide, panthenol, squalane, aloe, atau centella/cica.
- Untuk kulit sensitif: prioritaskan soothing dan barrier repair seperti centella/cica, panthenol, ceramide, aloe, hyaluronic acid. Hindari rekomendasi yang terlalu agresif.
- Untuk kulit kombinasi: pilih pelembap ringan yang menghidrasi tanpa terlalu oklusif, serta boleh mengandung niacinamide/hyaluronic acid.
- Untuk kulit normal/maintenance: jangan over-treatment. Prioritaskan basic skincare, hidrasi, skin barrier, dan sunscreen. Hindari AHA/BHA/retinol kecuali user eksplisit meminta.
- Untuk pori-pori besar: jelaskan bahwa pori tidak bisa dikecilkan permanen, hanya tampilan pori dapat dibantu dengan oil control dan unclog pores.
- Jangan memilih produk hanya karena masih berkaitan umum dengan kondisi kulit. Prioritaskan produk yang memiliki bahan aktif langsung sesuai masalah utama pengguna.
- Untuk bekas jerawat/PIH, produk tanpa bahan brightening aktif seperti vitamin C, niacinamide, alpha arbutin, azelaic acid, tranexamic acid, retinol, atau licorice hanya boleh direkomendasikan jika tidak ada pilihan lain.
- Untuk jerawat meradang, jangan prioritaskan produk anti-aging, whitening umum, atau barrier-only jika masih ada produk dengan salicylic acid, benzoyl peroxide, azelaic acid, sulfur, zinc, tea tree, centella, atau niacinamide.
- Untuk bruntusan/komedo, toner antioksidan atau soothing-only bukan prioritas utama kecuali tidak tersedia toner dengan BHA/AHA/salicylic acid.
- Untuk kulit sensitif, hindari bahan aktif konsentrasi tinggi seperti niacinamide 10%, vitamin C kuat, AHA/BHA, dan retinol sebagai rekomendasi utama kecuali konteks produk jelas menyatakan aman untuk kulit sensitif.
- Jika pengguna hamil, menyusui, atau sedang program hamil, jangan merekomendasikan produk yang mengandung retinol, retinal, retinoid, tretinoin, adapalene, atau tazarotene. Berikan alternatif non-retinoid dan sarankan konsultasi dokter.
- Jika pengguna mengalami iritasi, perih, merah, atau rusak barrier setelah exfoliating toner, jangan merekomendasikan AHA, BHA, PHA, salicylic acid, glycolic acid, lactic acid, retinol, atau tea tree sebagai pilihan utama. Fokus pada barrier repair dan soothing.
- Jangan menyatakan bahwa niacinamide, hyaluronic acid, atau cleanser biasa meningkatkan sensitivitas UV. Sunscreen tetap disarankan untuk perlindungan umum dan hiperpigmentasi.
- Jika pengguna hamil, menyusui, atau sedang program hamil, jangan merekomendasikan produk yang mengandung retinol, retinal, retinoid, tretinoin, adapalene, atau tazarotene. Berikan alternatif non-retinoid dan sarankan konsultasi dokter kandungan atau dokter kulit.
- Jika pengguna mengalami iritasi, perih, merah, atau rusak barrier setelah exfoliating toner, jangan merekomendasikan AHA, BHA, PHA, salicylic acid, glycolic acid, lactic acid, retinol, atau tea tree sebagai pilihan utama. Fokus pada barrier repair dan soothing.
- Jangan menyatakan bahwa hidrasi, moisturizer, hyaluronic acid, glycerin, ceramide, niacinamide, atau cleanser biasa meningkatkan sensitivitas terhadap UV. Sunscreen disarankan untuk perlindungan harian dan pencegahan hiperpigmentasi, bukan karena bahan tersebut membuat kulit fotosensitif.
- Gunakan dokumen bertanda [SUMBER: JURNAL DERMATOLOGI] sebagai dasar "Alasan Dermatologis (Jurnal)" pada setiap produk. Jika dokumen jurnal tidak cukup spesifik, tulis alasan umum yang aman berdasarkan jurnal dan jangan mengarang kutipan.
- Jika pengguna meminta produk yang "menyembuhkan jerawat permanen", "menghilangkan jerawat selamanya", atau klaim absolut sejenis, jelaskan di awal bahwa tidak ada skincare yang dapat menjamin kesembuhan permanen. Jika tetap memberi rekomendasi, posisikan produk hanya untuk membantu mengontrol jerawat, minyak, komedo, dan inflamasi.
""".strip()

    return f"""
Anda adalah GlowAI, asisten rekomendasi skincare berbasis data dan prinsip dermatologi dasar.

ATURAN UTAMA:
1. Jawab HANYA berdasarkan konteks yang diberikan.
2. Jangan mengarang nama produk, harga, kandungan, atau link produk.
3. Jangan membuat produk generik seperti "Vitamin C Serum" jika nama produk tidak ada di konteks.
4. Fokus hanya pada empat kategori produk: facial wash, toner, serum, dan moisturizer.
5. Jika pengguna menyebut kategori produk, gunakan hanya kategori tersebut.
6. Jika pengguna menyebut budget, patuhi budget secara ketat.
7. Tampilkan maksimal {jumlah_produk} produk yang paling relevan.
8. Jika data yang benar-benar sesuai kurang dari {jumlah_produk}, tampilkan yang tersedia saja lalu beri catatan keterbatasan data.
9. Pada bagian "Kandungan Kunci", tulis bahan aktif atau kandungan penting, bukan tekstur produk.
10. Pada bagian "Alasan Kesesuaian Produk", jelaskan singkat mengapa produk cocok dengan kebutuhan pengguna berdasarkan kandungan dan kategori produk.
11. Jangan memberi diagnosis medis profesional.
12. Jangan menyebut produk sebagai obat atau terapi medis definitif.
13. Jika link tidak tersedia di konteks, tulis "Link tidak tersedia dalam basis data", jangan mengarang link.
14. Jangan menulis harga Rp0. Jika harga tidak ada, tulis "Harga tidak tersedia".
15. Jika budget terdeteksi, hanya rekomendasikan produk yang harga eksplisitnya berada di bawah atau sama dengan budget.
16. Jika semua produk pada konteks melebihi budget, jangan rekomendasikan produk apa pun dan gunakan catatan keterbatasan data.
17. Jangan menampilkan proses berpikir internal, chain-of-thought, atau penalaran tersembunyi. Tampilkan jawaban final saja.
18. Gunakan bahasa ringkas. Analisis maksimal 2 kalimat, setiap alasan maksimal 1 kalimat per produk, dan Catatan maksimal 2 poin.
19. Jangan menampilkan produk yang memiliki status discontinued/tidak diproduksi lagi.
20. Jangan menyatakan produk aman untuk ibu hamil secara absolut. Untuk kehamilan/menyusui, sarankan konsultasi dengan dokter kandungan atau dokter kulit sebelum memakai bahan aktif.
21. Jangan menjanjikan hasil instan seperti putih dalam 3 hari, bekas jerawat hilang cepat, atau jerawat sembuh permanen. Berikan ekspektasi realistis bahwa perubahan kulit biasanya memerlukan beberapa minggu.
22. Jangan menggunakan klaim "menyembuhkan permanen". Untuk jerawat, jelaskan bahwa kondisi dapat dikontrol/dikelola dan dapat kambuh tergantung hormon, kebiasaan, dan faktor kulit.
23. Jika kandungan kunci tidak tersedia dalam basis data, jangan memberi alasan medis yang terlalu spesifik. Tulis bahwa alasan rekomendasi terbatas karena data kandungan tidak lengkap.
24. Buat satu bagian "Alasan Dermatologis Berdasarkan Jurnal" setelah daftar rekomendasi. Bagian ini harus merangkum prinsip dermatologi dari dokumen jurnal, bukan diulang pada setiap produk.

{clinical_rules}

{budget_rule}

{category_rule}

FORMAT JAWABAN:
### Analisis Kulit
[analisis singkat masalah utama pengguna]

### Rekomendasi
1. **Nama Produk**
- **Harga:** ...
- **Kandungan Kunci:** ...
- **Alasan Kesesuaian Produk:** ...
- **Link Produk:** ...

### Alasan Dermatologis Berdasarkan Jurnal
[maksimal 2-3 poin ringkas berdasarkan dokumen bertanda [SUMBER: JURNAL DERMATOLOGI]]

### Catatan Ringkas
[maksimal 2 poin catatan keamanan, batasan data, sunscreen, patch test, atau frekuensi pemakaian jika relevan]
""".strip()


def call_model(
    model_id: str,
    user_query: str,
    context: str,
    jumlah_produk: int,
    budget: Optional[float],
    category: Optional[str],
    concerns: List[str],
    use_openrouter: bool = False,
) -> str:
    prompt = build_prompt(jumlah_produk, budget, category, concerns)

    msg = f"""
PERTANYAAN PENGGUNA:
{user_query}

KONTEKS:
{context}
""".strip()

    active_client = openrouter_client if use_openrouter else groq_client

    res = active_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": msg},
        ],
        temperature=0.05,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    return res.choices[0].message.content.strip()


def generate_answer(
    user_query: str,
    context: str,
    jumlah_produk: int,
    budget: Optional[float],
    category: Optional[str],
    concerns: List[str],
) -> Tuple[str, str]:
    # Primary: Groq (llama-3.3-70b-versatile)
    try:
        answer = call_model(
            PRIMARY_MODEL_ID, user_query, context,
            jumlah_produk, budget, category, concerns,
            use_openrouter=False,
        )
        return answer, f"Groq / {PRIMARY_MODEL_ID}"
    except Exception as primary_err:
        # Backup: OpenRouter (openai/gpt-oss-120b)
        if openrouter_client is None:
            raise RuntimeError(
                f"Model utama gagal: {primary_err}. "
                "OPENROUTER_API_KEY tidak ditemukan di secrets.toml untuk fallback."
            )
        answer = call_model(
            BACKUP_MODEL_ID, user_query, context,
            jumlah_produk, budget, category, concerns,
            use_openrouter=True,
        )
        return answer, f"OpenRouter / {BACKUP_MODEL_ID} (fallback)"


def append_safety_notes(answer: str, concerns: List[str], category: Optional[str]) -> str:
    safety = generate_safety_notes(concerns, category)

    if not safety:
        return answer

    if "### Catatan" in answer or "### Catatan Ringkas" in answer:
        return answer.strip()

    return answer.strip() + "\n\n### Catatan Ringkas\n" + safety


# =====================================================
# 9. UI
# =====================================================
st.markdown(
    """
    <div class="info-box">
        <div class="info-title">Petunjuk Penggunaan</div>
        Tulis kebutuhan skincare secara spesifik, termasuk kategori produk, jumlah rekomendasi, atau budget jika perlu.<br><br>
        <div class="category-chips">
            <span class="chip">Facial Wash</span>
            <span class="chip">Toner</span>
            <span class="chip">Serum</span>
            <span class="chip">Moisturizer</span>
        </div>
        <ul>
            <li>Rekomendasikan 3 moisturizer untuk kulit kering yang terasa ketarik setelah mencuci muka.</li>
            <li>Kasih 2 serum untuk bekas jerawat dengan budget di bawah 50 ribu.</li>
            <li>Berikan 4 toner untuk bruntusan di dahi dan pipi.</li>
            <li>Carikan facial wash untuk kulit berminyak dan mudah berjerawat.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "Masukkan kebutuhan skincare Anda",
    placeholder="Contoh: Rekomendasikan 3 serum untuk bekas jerawat dengan budget di bawah 100 ribu.",
    height=200,
)

if st.button("Analisis dan Rekomendasikan", type="primary"):
    ok, msg = validate_input(user_input)

    if not ok:
        st.error(msg)
        st.stop()

    jumlah_produk = extract_requested_count(user_input)
    budget = extract_budget(user_input)
    category = detect_category(user_input)
    concerns = detect_skin_concerns(user_input)

    with st.spinner("Mencari dokumen relevan..."):
        docs = retrieve(user_input, k=TOPK_CONTEXT)

    product_count = count_product_docs(docs)
    journal_count = count_journal_docs(docs)

    if product_count == 0:
        st.markdown(
            "<div class='warning-box'>Tidak ditemukan produk yang memenuhi kategori dan/atau budget yang diminta.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(fallback_response(jumlah_produk, category, budget, concerns))
        st.stop()

    context = build_context(docs)

    with st.spinner("Menyusun rekomendasi..."):
        try:
            answer, model_used = generate_answer(
                user_input,
                context,
                jumlah_produk,
                budget,
                category,
                concerns,
            )
        except Exception as e:
            st.error(f"Gagal generate jawaban: {e}")
            st.stop()

    if not valid_output(answer) or answer_violates_budget(answer, budget) or answer_looks_incomplete(answer):
        answer = fallback_response(jumlah_produk, category, budget, concerns)

    answer = append_safety_notes(answer, concerns, category)

    st.markdown(f"<div class='result-box'>{answer}</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='meta-strip' style='margin-top: 0.75rem;'>"
        f"Model: {model_used} &nbsp;&middot;&nbsp; "
        f"Rekomendasi: {jumlah_produk} &nbsp;&middot;&nbsp; "
        f"Kategori: {category if category else 'Tidak disebutkan'} &nbsp;&middot;&nbsp; "
        f"Budget: {format_rp(budget) if budget is not None else 'Tidak disebutkan'} &nbsp;&middot;&nbsp; "
        f"Kondisi: {', '.join(concerns) if concerns else 'Tidak spesifik'} &nbsp;&middot;&nbsp; "
        f"Dokumen: {len(docs)} &nbsp;&middot;&nbsp; "
        f"Jurnal: {journal_count}"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Transparansi Retrieval (Explainable RAG)"):
        st.markdown("**Informasi Retrieval**")
        st.write(f"- **Kategori terdeteksi:** {category if category else 'Tidak disebutkan'}")
        st.write(f"- **Budget terdeteksi:** {format_rp(budget) if budget is not None else 'Tidak disebutkan'}")
        st.write(f"- **Kondisi kulit terdeteksi:** {', '.join(concerns) if concerns else 'Tidak spesifik'}")
        st.write(f"- **Jumlah rekomendasi diminta:** {jumlah_produk}")
        st.write(f"- **Initial retrieval:** {INITIAL_RETRIEVAL}")
        st.write(f"- **Top-k context ke LLM:** {TOPK_CONTEXT}")
        st.write(f"- **Minimum similarity score:** {MIN_SIMILARITY_SCORE}")
        st.write(f"- **Jumlah dokumen konteks yang digunakan:** {len(docs)}")
        st.write(f"- **Jumlah dokumen produk valid:** {product_count}")
        st.write(f"- **Jumlah dokumen jurnal:** {journal_count}")

        st.markdown("---")
        st.markdown("**Detail Dokumen yang Digunakan**")

        for i, d in enumerate(docs, 1):
            explanation = []

            if category and d.get("category") and category == d["category"]:
                explanation.append("kategori sesuai")

            if budget and d.get("price") is not None and d["price"] <= budget:
                explanation.append("sesuai budget")

            if d.get("source") == "jurnal":
                explanation.append("referensi medis")

            if d.get("source") == "produk":
                explanation.append("relevansi klinis berdasarkan kondisi kulit")

            if not explanation:
                explanation.append("relevansi semantik umum")

            explanation_text = ", ".join(explanation)

            meta_lines = [
                f"**Dokumen {i}**",
                f"- Base score: `{d.get('base_score', 0):.3f}`",
                f"- Final score: `{d.get('score', 0):.3f}`",
                f"- Sumber: `{d.get('source', '-')}`",
            ]

            if d.get("product_name"):
                meta_lines.append(f"- Produk: `{d['product_name']}`")

            if d.get("brand"):
                meta_lines.append(f"- Brand: `{d['brand']}`")

            if d.get("category"):
                meta_lines.append(f"- Kategori: `{d['category']}`")

            if d.get("topic"):
                meta_lines.append(f"- Topik jurnal: `{d['topic']}`")

            if d.get("source_file"):
                meta_lines.append(f"- File jurnal: `{d['source_file']}`")

            if d.get("page"):
                meta_lines.append(f"- Halaman: `{d['page']}`")

            raw_price = d.get("raw_price", "")

            if d.get("price") is not None:
                meta_lines.append(f"- Harga: `{format_rp(d['price'])}`")
            elif raw_price:
                meta_lines.append(f"- Harga: `{raw_price}`")

            if d.get("url"):
                meta_lines.append(f"- Link: {d['url']}")

            meta_lines.append(f"- Alasan dipilih: **{explanation_text}**")

            st.markdown("\n".join(meta_lines))
            st.caption(d["text"][:350] + ("..." if len(d["text"]) > 350 else ""))
            st.markdown("---")

st.markdown("<div class='footer-note'>GlowAI Research Project &mdash; 2025</div>", unsafe_allow_html=True)