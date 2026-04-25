# CS4241 Manual RAG Chatbot — Configuration
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
RUNS = EXPERIMENTS / "runs"

CSV_PATH = DATA_RAW / "Ghana_Election_Result.csv"
PDF_PATH = DATA_RAW / "2025-Budget-Statement.pdf"
CHUNKS_PATH = DATA_PROCESSED / "chunks.json"
INDEX_PATH = DATA_PROCESSED / "index.npz"
SESSION_PATH = DATA_PROCESSED / "session.json"

# ── Data sources ──────────────────────────────────────────────────────────────
CSV_URL = "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
PDF_URL = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"

# ── Chunking ──────────────────────────────────────────────────────────────────
# Three strategies compared in exp_chunking.py
CHUNK_FIXED_SIZE = 500       # chars, baseline
CHUNK_FIXED_OVERLAP = 50     # chars

CHUNK_SENTENCE_TOKENS = 400  # approx token budget per sentence-aware chunk
CHUNK_SENTENCE_OVERLAP = 50  # tokens of overlap

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, runs on CPU
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIM = 384

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5                    # chunks returned per query
MIN_SCORE = 0.20             # drop chunks below this combined score
HYBRID_ALPHA = 0.7           # weight for vector score; (1-α) for BM25

# ── Novel feature (PART G) ────────────────────────────────────────────────────
DOMAIN_BOOST_BETA = 0.10     # score bonus when chunk intent matches query intent
MEMORY_MAX_TURNS = 5         # ring buffer size for session memory

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

# ── Context window management (PART C) ───────────────────────────────────────
MAX_CONTEXT_CHARS = 12_000   # hard cap on total context fed to LLM
MAX_CHUNK_CHARS = 3_000      # per-chunk truncation limit
