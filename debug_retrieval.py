"""
סקריפט אבחון מלא – מראה את כל שלבי הפייפליין בדיוק כמו האפליקציה:
  1. ספירת chunks  2. בדיקת Cohere
  3. תרגום + Query Expansion
  4. Hybrid Search (top_k=50) + מיזוג
  5. Reranking (Cohere אם פעיל, אחרת RRF)
  6. הצגת Top-5 עמודים עם ציוני Cohere / RRF
הרץ: python debug_retrieval.py
"""
import os
import anthropic
import chromadb
from dotenv import load_dotenv

load_dotenv()  # חיוני – טוען ANTHROPIC_API_KEY ו-COHERE_API_KEY מ-.env

from rag import hybrid_search

# cohere בדיקה עצמאית – לא תלוי בייבוא פרטי מ-rag.py
try:
    import cohere as _cohere
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False

# ── הגדרות ──────────────────────────────────────────────────────────────────
QUERY_HE      = "מה הנפח של המאגר העליון והמאגר התחתון"
SOURCE_FILTER = None  # אפשר להגביל: "spec.pdf"

# ── 1. ספירת Chunks ─────────────────────────────────────────────────────────
לקוח_chroma = chromadb.PersistentClient(path="chroma_db")
אוסף = לקוח_chroma.get_or_create_collection(name="pdf_collection")
סה_כ_chunks = אוסף.count()

print(f"\n{'='*65}")
print(f"📦 Chunks ב-ChromaDB: {סה_כ_chunks}")
if סה_כ_chunks == 0:
    print("⛔ המסד ריק! הרץ: python rag.py")
    raise SystemExit(1)

# ── 2. בדיקת Cohere ──────────────────────────────────────────────────────────
_cohere_key   = os.environ.get("COHERE_API_KEY")
reranker_active = _COHERE_AVAILABLE and bool(_cohere_key)

print(f"[CHECK] Cohere package installed : {'YES' if _COHERE_AVAILABLE else 'NO'}")
print(f"[CHECK] Cohere API Key detected  : {'YES' if _cohere_key else 'NO'}")
print(f"[CHECK] Reranker Status          : {'ACTIVE (Using Cohere)' if reranker_active else 'INACTIVE (Fallback to RRF)'}")
print(f"{'='*65}\n")

# ── 3. תרגום + Query Expansion ───────────────────────────────────────────────
לקוח_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

שאלה_באנגלית = לקוח_anthropic.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": (
        "Translate the following question to English. "
        "Return ONLY the translated question, no explanation.\n\n"
        f"Question: {QUERY_HE}"
    )}],
).content[0].text.strip()

גרסאות_נוספות = [
    ש.strip()
    for ש in לקוח_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": (
            "Generate 2 alternative English search queries for the following question. "
            "Query 1: use abbreviated technical units (e.g. 'mio m3', 'MCM', 'million m3', 'Mm3'). "
            "Query 2: use expanded descriptive terms (e.g. 'million cubic meters', 'storage capacity', 'total volume'). "
            "Return ONLY the 2 queries, one per line, no numbering or explanation.\n\n"
            f"Original query: {שאלה_באנגלית}"
        )}],
    ).content[0].text.strip().splitlines()
    if ש.strip()
][:2]

כל_שאילתות = [שאלה_באנגלית] + גרסאות_נוספות
print("🔤 שאילתות לחיפוש:")
for i, ש in enumerate(כל_שאילתות, 1):
    print(f"  [{i}] {ש}")
print()

# ── 4. Hybrid Search (top_k=40) + מיזוג ─────────────────────────────────────
מיטב_לפי_עמוד: dict[tuple, dict] = {}

for שאילתה_רחבה in כל_שאילתות:
    _, מקורות_q, ציונים_q, עמודים_q, פולים_q = hybrid_search(
        question_en=שאילתה_רחבה,
        collection=אוסף,
        filter_source=SOURCE_FILTER,
        n_results=50,
    )
    for מקור, ציון, עמוד, פול in zip(מקורות_q, ציונים_q, עמודים_q, פולים_q):
        מפתח = (מקור, עמוד)
        if מפתח not in מיטב_לפי_עמוד or ציון > מיטב_לפי_עמוד[מפתח]["ציון"]:
            מיטב_לפי_עמוד[מפתח] = {
                "ציון": ציון, "full": פול, "source": מקור, "page": עמוד
            }

ממוין_ראשוני = sorted(מיטב_לפי_עמוד.values(), key=lambda x: x["ציון"], reverse=True)
print(f"📊 עמודים ייחודיים אחרי מיזוג: {len(ממוין_ראשוני)}\n")

# ── 5. Reranking ──────────────────────────────────────────────────────────────
MAX_PAGES = 10       # עמודים שמגיעים ל-Claude
RERANK_WINDOW = 100  # עמודים מקסימליים שנשלחים ל-Cohere

if reranker_active:
    co = _cohere.ClientV2(api_key=_cohere_key)
    מועמדים_לrerank = ממוין_ראשוני[:RERANK_WINDOW]
    תגובת_rerank = co.rerank(
        model="rerank-v3.5",
        query=שאלה_באנגלית,
        documents=[item["full"][:1500] for item in מועמדים_לrerank],
        top_n=min(MAX_PAGES, len(מועמדים_לrerank)),
    )
    ממוין = [
        {**מועמדים_לrerank[r.index], "cohere_score": r.relevance_score}
        for r in תגובת_rerank.results
    ]
else:
    ממוין = ממוין_ראשוני[:MAX_PAGES]

# ── 6. הצגת Top-5 עמודים ────────────────────────────────────────────────────
print(f"{'='*65}")
print(f"🔍 Top-15 עמודים – טקסט גולמי מלא")
print(f"{'='*65}\n")

for i, item in enumerate(ממוין[:15], start=1):
    if reranker_active:
        ציון_תצוגה = f"Cohere={item['cohere_score']:.4f}"
    else:
        ציון_תצוגה = f"RRF={item['ציון']:.6f}"
    print(f"── [{i}] {ציון_תצוגה} | עמוד {item['page']} | {item['source']}")
    print(item["full"])
    print()

print(f"{'='*65}\n")
