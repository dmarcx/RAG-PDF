"""
debug_page.py – אבחון עמוד ספציפי:
  1. טקסט גולמי של העמוד כפי ש-pdfplumber רואה אותו
  2. חיפוש ישיר ב-ChromaDB (ללא LLM) עבור מחרוזת ספציפית
  3. כל ה-Chunks של העמוד כפי שנשמרו ב-DB
הרץ: python debug_page.py
"""
import os
import pdfplumber
import chromadb
from dotenv import load_dotenv

load_dotenv()

from rag import _page_to_text, _extract_section_header

# ── הגדרות ──────────────────────────────────────────────────────────────────
PAGE_NUMBER = 10        # עמוד לבדיקה (1-based)
SEARCH_STR  = "1.18"   # מחרוזת לחיפוש ישיר ב-ChromaDB
PDF_FILE    = None      # None = ייקח את הראשון בתיקיית pdfs/ אוטומטית
                        # אפשר לציין: PDF_FILE = "spec.pdf"

# ── מציאת קובץ PDF ─────────────────────────────────────────────────────────
pdfs_dir = "pdfs"
if PDF_FILE is None:
    כל_pdfs = sorted(f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf"))
    if not כל_pdfs:
        print("⛔ אין קבצי PDF בתיקיית pdfs/")
        raise SystemExit(1)
    PDF_FILE = כל_pdfs[0]

נתיב_pdf = os.path.join(pdfs_dir, PDF_FILE)
print(f"\n📄 PDF: {PDF_FILE} | עמוד נבדק: {PAGE_NUMBER}")
print(f"{'='*65}\n")

# ── 1. טקסט גולמי מ-pdfplumber ───────────────────────────────────────────────
print("═"*65)
print(f"📋 1. טקסט גולמי – pdfplumber.extract_text() (עמוד {PAGE_NUMBER})")
print("═"*65)

with pdfplumber.open(נתיב_pdf) as pdf:
    if PAGE_NUMBER > len(pdf.pages):
        print(f"⛔ {PDF_FILE} מכיל רק {len(pdf.pages)} עמודים")
        raise SystemExit(1)
    עמוד_obj = pdf.pages[PAGE_NUMBER - 1]
    טקסט_גולמי     = עמוד_obj.extract_text() or ""
    טקסט_מעובד     = _page_to_text(עמוד_obj)   # כולל המרת טבלאות

# הצגה מפורמטת
print(טקסט_גולמי or "(ריק)")
print()

# בדיקה אם המחרוזת קיימת
if SEARCH_STR in טקסט_גולמי:
    print(f"✅ '{SEARCH_STR}' נמצא ב-extract_text()")
    for שורה in טקסט_גולמי.splitlines():
        if SEARCH_STR in שורה:
            print(f"   → {שורה!r}")
else:
    print(f"❌ '{SEARCH_STR}' לא נמצא ב-extract_text()!")

# הצגת הטקסט לאחר המרת טבלאות (_page_to_text)
print()
print("── לאחר המרת טבלאות (_page_to_text):")
if טקסט_מעובד != טקסט_גולמי:
    print(טקסט_מעובד[:1500] or "(ריק)")
    if SEARCH_STR in טקסט_מעובד and SEARCH_STR not in טקסט_גולמי:
        print(f"ℹ️  '{SEARCH_STR}' מופיע רק לאחר המרת טבלאות")
else:
    print("(זהה ל-extract_text – אין טבלאות בעמוד זה)")

# כותרת שחולצה
כותרת = _extract_section_header(טקסט_מעובד)
print(f"\n🏷️  _extract_section_header → {כותרת!r}")
print()

# ── 2. חיפוש ישיר ב-ChromaDB ─────────────────────────────────────────────────
print("═"*65)
print(f"🔍 2. חיפוש ישיר ב-ChromaDB עבור: '{SEARCH_STR}'")
print("═"*65)

לקוח_chroma = chromadb.PersistentClient(path="chroma_db")
אוסף = לקוח_chroma.get_or_create_collection(name="pdf_collection")

תוצאות_חיפוש = אוסף.get(
    where_document={"$contains": SEARCH_STR},
    include=["documents", "metadatas"],
)

if not תוצאות_חיפוש["documents"]:
    print(f"❌ אין chunks ב-ChromaDB שמכילים '{SEARCH_STR}'")
    print("   ייתכן: אינדוקס לא רץ, או המחרוזת בפורמט שונה (רווחים, Unicode)")
else:
    print(f"✅ נמצאו {len(תוצאות_חיפוש['documents'])} chunks")
    for i, (מסמך, מטא) in enumerate(
        zip(תוצאות_חיפוש["documents"], תוצאות_חיפוש["metadatas"]), 1
    ):
        idx   = מסמך.find(SEARCH_STR)
        start = max(0, idx - 100)
        end   = min(len(מסמך), idx + 100)
        print(f"\n  [{i}] עמוד {מטא.get('page_number')} | {מטא.get('source')}")
        print(f"       ...{מסמך[start:end]}...")
print()

# ── 3. Chunks של עמוד PAGE_NUMBER ─────────────────────────────────────────────
print("═"*65)
print(f"🧩 3. כל ה-Chunks של עמוד {PAGE_NUMBER} ({PDF_FILE}) ב-ChromaDB")
print("═"*65)

תוצאות_עמוד = אוסף.get(
    where={"$and": [{"page_number": {"$eq": PAGE_NUMBER}}, {"source": {"$eq": PDF_FILE}}]},
    include=["documents", "metadatas"],
)

if not תוצאות_עמוד["documents"]:
    print(f"❌ אין chunks לעמוד {PAGE_NUMBER} ב-ChromaDB")
    print("   ייתכן: עמוד ריק, אינדוקס לא רץ, או שם הקובץ שונה")
    # הדפסת שמות הקבצים הקיימים ב-DB
    כל_מטא = אוסף.get(include=["metadatas"])["metadatas"]
    שמות_ייחודיים = sorted({m["source"] for m in כל_מטא})
    print(f"   קבצים ב-DB: {שמות_ייחודיים}")
else:
    print(f"📊 {len(תוצאות_עמוד['documents'])} chunks לעמוד {PAGE_NUMBER}\n")
    for i, (מסמך, מטא) in enumerate(
        zip(תוצאות_עמוד["documents"], תוצאות_עמוד["metadatas"]), 1
    ):
        serial = מטא.get("chunk_serial", "?")
        יש_מחרוזת = "✅" if SEARCH_STR in מסמך else "  "
        print(f"── Chunk {i} (serial={serial}) {יש_מחרוזת}")
        print(מסמך)
        print()

print(f"{'='*65}\n")
