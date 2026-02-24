"""
סקריפט בדיקה חד-פעמי: קורא עמוד 10 ומדפיס טקסט + טבלאות.
הרץ: python test_pdf.py
"""
import pdfplumber

PDF_PATH = r"pdfs\MPP-AFR-PMG-SCM-GEN-REP 00058.00 Civil Works Outline Design Report.pdf"
PAGE_NUM = 10  # אינדקס 0-based → עמוד 11 בקובץ? לא – כאן נשתמש בעמוד ה-10 (index 9)

with pdfplumber.open(PDF_PATH) as pdf:
    total = len(pdf.pages)
    print(f"סה״כ עמודים בקובץ: {total}")
    print(f"בודק עמוד {PAGE_NUM} (index {PAGE_NUM - 1})\n")

    if PAGE_NUM > total:
        print(f"שגיאה: הקובץ מכיל רק {total} עמודים.")
        exit(1)

    page = pdf.pages[PAGE_NUM - 1]  # 1-based → 0-based

    # ── טקסט גולמי ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("📄 PLAIN TEXT (extract_text)")
    print("=" * 70)
    טקסט = page.extract_text()
    print(טקסט if טקסט else "(אין טקסט בעמוד זה)")

    # ── טבלאות ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 TABLES (extract_tables)")
    print("=" * 70)
    טבלאות = page.extract_tables()

    if not טבלאות:
        print("(לא נמצאו טבלאות בעמוד זה)")
    else:
        print(f"נמצאו {len(טבלאות)} טבלאות:\n")
        for i, טבלאה in enumerate(טבלאות, start=1):
            print(f"── טבלאה {i} ──────────────────────────────────────")
            כותרות = [str(תא or "").strip() for תא in (טבלאה[0] or [])]
            print("כותרות:", " | ".join(כותרות))
            print()
            for j, שורה in enumerate(טבלאה[1:], start=1):
                זוגות = [
                    f"{(כותרות[k] or '?')}: {(str(תא or '')).strip()}"
                    for k, תא in enumerate(שורה)
                    if k < len(כותרות)
                ]
                print(f"  [{j:02d}] " + " | ".join(זוגות))
            print()

    # ── תצוגת _page_to_text (מה שנשמר ב-ChromaDB בפועל) ────────────────────
    print("=" * 70)
    print("💾 מה שנשמר ב-ChromaDB (_page_to_text)")
    print("=" * 70)
    from rag import _page_to_text
    print(_page_to_text(page))

# ── חיפוש ChromaDB ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🔎 חיפוש ChromaDB")
print("=" * 70)

QUESTION_HE = "מה הנפח של המאגר העליון והמאגר התחתון"
QUESTION_EN = "What is the volume of the upper reservoir and the lower reservoir"
TARGET_PAGE = PAGE_NUM  # בודקים אם עמוד זה מופיע בתוצאות

import chromadb
לקוח = chromadb.PersistentClient(path="chroma_db")
אוסף = לקוח.get_or_create_collection(name="pdf_collection")

print(f"שאלה (עברית): {QUESTION_HE}")
print(f"שאלה (אנגלית): {QUESTION_EN}\n")

for label, query in [("עברית", QUESTION_HE), ("אנגלית", QUESTION_EN)]:
    תוצאות = אוסף.query(query_texts=[query], n_results=15)
    מסמכים = תוצאות["documents"][0]
    מטא     = תוצאות["metadatas"][0]
    מרחקים  = תוצאות["distances"][0]

    print(f"── חיפוש ב{label} ──────────────────────────────────────")
    נמצא = False
    for i, (מסמך, מטא_פריט, מרחק) in enumerate(zip(מסמכים, מטא, מרחקים), start=1):
        chunk_idx = מטא_פריט.get("chunk_index", "?")
        מקור     = מטא_פריט.get("source", "?")
        # chunk_index = page_number - 1
        עמוד_בפועל = chunk_idx + 1 if isinstance(chunk_idx, int) else "?"
        סמן = " ◄◄◄ עמוד 10!" if עמוד_בפועל == TARGET_PAGE else ""
        תצוגה = מסמך[:120].replace("\n", " ")
        print(f"  [{i:02d}] dist={מרחק:.4f} | עמוד {עמוד_בפועל} | {מקור[:40]}{סמן}")
        print(f"       {תצוגה}...")
        if עמוד_בפועל == TARGET_PAGE:
            נמצא = True
    if not נמצא:
        print(f"  ⚠️  עמוד {TARGET_PAGE} לא מופיע ב-15 התוצאות הראשונות!")
    print()
