import os
import re
import shutil
import anthropic
import chromadb
import pdfplumber
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# cohere אופציונלי – נדרש ל-Reranking (pip install cohere, ו-COHERE_API_KEY ב-.env)
try:
    import cohere as _cohere
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False

load_dotenv()  # טוען את משתני הסביבה מקובץ .env


def _page_to_text(page) -> str:
    """
    ממיר עמוד pdfplumber לטקסט:
    - טבלאות: עמודה: ערך | עמודה: ערך
    - שאר הטקסט: כרגיל
    """
    חלקים = []

    # זיהוי טבלאות בענוד
    טבלאות = page.extract_tables()
    טקסט_העמוד = page.extract_text() or ""

    if טבלאות:
        # שמירת הטקסט הרגיל לפני הטבלאות
        if טקסט_העמוד.strip():
            חלקים.append(טקסט_העמוד)

        for טבלאה in טבלאות:
            שורות_מומרות = []

            # השורה הראשונה היא הכותרת
            כותרות = [תא if תא else "" for תא in (טבלאה[0] or [])]

            for שורה in טבלאה[1:]:
                # מדלג שורות ריקות
                if not any(תא for תא in שורה if תא):
                    continue
                זוגות = [
                    f"{(כותרות[i] or '').strip()}: {(תא or '').strip()}"
                    for i, תא in enumerate(שורה)
                    if i < len(כותרות)
                ]
                שורות_מומרות.append(" | ".join(זוגות))

            if שורות_מומרות:
                חלקים.append("[TABLE]\n" + "\n".join(שורות_מומרות))
    else:
        if טקסט_העמוד.strip():
            חלקים.append(טקסט_העמוד)

    return "\n".join(חלקים)


def load_pdf(file_path: str) -> str:
    """קורא קובץ PDF ומחזיר את כל הטקסט כמחרוזת, כולל טבלאות מומרות."""
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(_page_to_text(page) for page in pdf.pages)


def load_pdf_pages(file_path: str):
    """גנרטור שמחזיר טקסט עמוד-עמוד – חוסך זיכרון לקבצים ענקיים."""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            yield _page_to_text(page)


def count_pdf_pages(file_path: str) -> int:
    """מחזיר את מספר העמודים בקובץ PDF."""
    with pdfplumber.open(file_path) as pdf:
        return len(pdf.pages)


def save_to_chromadb_batch(chunks: list[dict]) -> None:
    """שומר אצווה של chunks ל-ChromaDB - כולל page_number + full_page_content במטא."""
    if not chunks:
        return

    לקוח = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח.get_or_create_collection(name="pdf_collection")

    מזהים  = [f"{c['source']}__p{c['page_number']}__s{c['chunk_serial']}" for c in chunks]
    מסמכים = [c["text"] for c in chunks]
    מטא    = [
        {
            "source":           c["source"],
            "page_number":      c["page_number"],
            "chunk_serial":     c["chunk_serial"],
            "full_page_content": c["full_page_content"],
        }
        for c in chunks
    ]

    אוסף.add(ids=מזהים, documents=מסמכים, metadatas=מטא)


def clear_chroma_db() -> None:
    """
    מאפס את אוסף pdf_collection ב-ChromaDB דרך ה-API.
    משתמש ב-API ולא במחיקת קבצים – עובד גם כאשר הקובץ תפוס (Windows).
    """
    לקוח = chromadb.PersistentClient(path="chroma_db")
    try:
        לקוח.delete_collection(name="pdf_collection")
        print("האוסף pdf_collection נמחק בהצלחה.")
    except Exception:
        print("האוסף לא נמצא – אין מה למחוק.")


def _extract_section_header(text: str) -> str | None:
    """
    מחלץ את הכותרת הממוספרת העמוקה ביותר מתוך טקסט העמוד.
    למשל: '6.2.3 Design description' יועדף על '6.2 Overview' (עמוק יותר = ספציפי יותר).
    כותרת עברית תשמש גיבוי בלבד אם לא נמצאה כותרת ממוספרת.
    """
    כותרת_נבחרת = None
    עומק_מקסימלי = -1

    for שורה in text.splitlines():
        שורה = שורה.strip()
        if not שורה or len(שורה) > 100:
            continue
        # כותרת ממוספרת: "6.2" / "6.2.3" – מגביל ל-2 ספרות ראשיות + אות גדולה, מונע "738.00 m3"
        m = re.match(r'^([1-9]\d{0,1}(?:\.\d{1,2}){1,2})\s+[A-Z]', שורה)
        if m:
            עומק = m.group(1).count('.')
            if עומק > עומק_מקסימלי:
                עומק_מקסימלי = עומק
                כותרת_נבחרת = שורה
        # מילות מפתח עבריות – גיבוי בלבד אם אין כותרת ממוספרת
        elif כותרת_נבחרת is None:
            if re.match(r'^(פרק|סעיף|פסקה|חלק|נספח|תוספת|נוהל|הגדרות|מבוא)\b', שורה):
                כותרת_נבחרת = שורה

    return כותרת_נבחרת


def _generate_chunk_context(
    לקוח_anthropic,
    source_name: str,
    page_text: str,
) -> str:
    """
    Contextual Retrieval: מייצר משפט הקשר קצר (עד 50 מילים) לכל עמוד.
    המשפט מוסף לתחילת כל Chunk מאותו עמוד לפני יצירת הוקטור ב-ChromaDB.
    מייצר פעם אחת לעמוד (לא לכל chunk) – חיסכון ב-N-1 קריאות API לעמוד.
    """
    try:
        תגובה = לקוח_anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Document: {source_name}\n"
                    f"Page text (first 600 chars): {page_text[:600]}\n\n"
                    "Write ONE sentence (max 50 words) describing what this page covers. "
                    "Include: document name, section topic, key technical terms "
                    "(units like mio m3/MCM, numeric values, categories). "
                    "Return ONLY the sentence."
                ),
            }],
        )
        return תגובה.content[0].text.strip()
    except Exception:
        return ""  # fallback – הchunk יישמר ללא הקשר


def process_large_pdf(
    file_path: str,
    source_name: str,
    chunk_size: int = 500,
    overlap: int = 100,
    batch_size: int = 200,
    progress_callback=None,
) -> int:
    """
    Parent Document Retrieval indexing:
    - כל עמוד מחולק ל-chunks קטנים (chunk_size תווים, overlap חפיפה)
    - במטא של כל chunk נשמר הטקסט המלא של העמוד (full_page_content)
    - בשלב השאילה יישלח ל-Claude העמוד המלא (לא רק ה-chunk)
    """
    # אתחול לקוח Anthropic לשימוש ב-Contextual Retrieval (פעם אחת לאינדוקס)
    לקוח_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    סה_כ_עמודים = count_pdf_pages(file_path)
    כל_החלקים: list[dict] = []
    סה_כ_chunks = 0
    סידורי_גלובלי = 0  # chunk serial רץ עבור מזהה ייחודי

    for מספר_עמוד, טקסט_עמוד in enumerate(load_pdf_pages(file_path), start=1):
        טקסט_מלא = טקסט_עמוד.strip()
        if not טקסט_מלא:
            if progress_callback:
                progress_callback(מספר_עמוד, סה_כ_עמודים)
            continue

        # בניית prefix: שם קובץ + כותרת סעיף (אם קיימת) – משפר התאמת BM25 וסמנטיקה
        כותרת_סעיף = _extract_section_header(טקסט_מלא)
        prefix = f"[מקור: {source_name}"
        if כותרת_סעיף:
            prefix += f" | סעיף: {כותרת_סעיף}"
        prefix += "]\n"

        # Contextual Retrieval: קריאת LLM פעם אחת לעמוד – משותף לכל ה-chunks שלו
        הקשר_עמוד = _generate_chunk_context(לקוח_anthropic, source_name, טקסט_מלא)

        # חיתוך העמוד ל-chunks קטנים
        עמדה_ב_טקסט = 0
        while עמדה_ב_טקסט < len(טקסט_מלא):
            חלק_גולמי = טקסט_מלא[עמדה_ב_טקסט:עמדה_ב_טקסט + chunk_size]
            # הקשר LLM → prefix → גוף הchunk – full_page_content נשאר נקי
            חלק = (f"{הקשר_עמוד}\n" if הקשר_עמוד else "") + prefix + חלק_גולמי
            כל_החלקים.append({
                "source":            source_name,
                "page_number":       מספר_עמוד,
                "chunk_serial":      סידורי_גלובלי,
                "text":              חלק,
                "full_page_content": טקסט_מלא,
            })
            סידורי_גלובלי += 1
            עמדה_ב_טקסט += chunk_size - overlap

        # שמירה באצווה
        if len(כל_החלקים) >= batch_size:
            save_to_chromadb_batch(כל_החלקים)
            סה_כ_chunks += len(כל_החלקים)
            כל_החלקים = []

        if progress_callback:
            progress_callback(מספר_עמוד, סה_כ_עמודים)

    if כל_החלקים:
        save_to_chromadb_batch(כל_החלקים)
        סה_כ_chunks += len(כל_החלקים)

    return סה_כ_chunks


def load_multiple_pdfs(folder_path: str) -> list[dict]:
    """
    קורא את כל קבצי ה-PDF בתיקייה.
    מחזיר רשימה של מילונים עם שני שדות:
      - source: שם הקובץ
      - text: הטקסט המלא
    """
    תוצאות = []

    # בודק שהתיקייה קיימת
    if not os.path.isdir(folder_path):
        print(f"תיקייה לא נמצאה: {folder_path}")
        return תוצאות

    for שם_קובץ in os.listdir(folder_path):
        # מתעלם מקבצים שאינם PDF
        if not שם_קובץ.lower().endswith(".pdf"):
            continue

        נתיב_מלא = os.path.join(folder_path, שם_קובץ)
        print(f"טוען: {שם_קובץ}")
        טקסט = load_pdf(נתיב_מלא)
        תוצאות.append({"source": שם_קובץ, "text": טקסט})

    return תוצאות


def split_text(text: str, source_name: str, chunk_size: int = 1500, overlap: int = 200) -> list[dict]:
    """
    מפצל טקסט לחלקים (chunks) עם חפיפה בין חלקים סמוכים.
    כל חלק מוחזר כמילון עם:
      - source: שם המקור (שם הקובץ)
      - chunk_index: מספר סידורי של החלק
      - text: תוכן החלק
    """
    חלקים = []
    מיקום = 0
    אינדקס = 0

    while מיקום < len(text):
        # חותך חלק בגודל chunk_size מהמיקום הנוכחי
        סוף = מיקום + chunk_size
        חלק = text[מיקום:סוף]

        # שומר רק חלקים שאינם ריקים לחלוטין
        if חלק.strip():
            חלקים.append({
                "source": source_name,
                "chunk_index": אינדקס,
                "text": חלק,
            })
            אינדקס += 1

        # מקדם את המיקום תוך שמירה על חפיפה
        מיקום += chunk_size - overlap

    return חלקים


def save_to_chromadb(chunks: list[dict]) -> None:
    """
    שומר חלקים ל-ChromaDB באוסף 'pdf_collection'.
    מניח שהחלקים שמועברים הם רק קבצים חדשים (לא קיימים).
    """
    # פותח את מסד הנתונים המקומי
    לקוח = chromadb.PersistentClient(path="chroma_db")

    # מקבל או יוצר את האוסף מבלי למחוק נתונים קיימים
    אוסף = לקוח.get_or_create_collection(name="pdf_collection")

    # מכין את הנתונים להכנסה מרוכזת
    מזהים = []
    מסמכים = []
    מטא_דאטה = []

    for חלק in chunks:
        מזהה = f"{חלק['source']}__chunk_{חלק['chunk_index']}"
        מזהים.append(מזהה)
        מסמכים.append(חלק["text"])
        מטא_דאטה.append({"source": חלק["source"], "chunk_index": חלק["chunk_index"]})

    # שומר את כל החלקים בבת אחת
    אוסף.add(ids=מזהים, documents=מסמכים, metadatas=מטא_דאטה)


def get_existing_sources() -> set:
    """
    מחזיר קבוצה של שמות הקבצים שכבר נידקסו ב-ChromaDB.
    משתמשת במטאדאטה של הרשומות הקיימות.
    """
    לקוח = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח.get_or_create_collection(name="pdf_collection")

    # משיג את כל המטאדאטה הקיימת במסד הנתונים
    תוצאות = אוסף.get(include=["metadatas"])

    # אוסף שמות ייחודיים לקבוצה
    שמות_קיימים = {m["source"] for m in תוצאות["metadatas"]}
    return שמות_קיימים


def list_sources() -> None:
    """
    מדפיסה כמה chunks יש מכל קובץ ב-ChromaDB.
    שימושית לבדיקה שכל המסמכים נטענו בהצלחה.
    """
    לקוח = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח.get_or_create_collection(name="pdf_collection")

    # שולף את כל המטאדאטה מהאוסף
    תוצאות = אוסף.get(include=["metadatas"])

    # סופר chunks לפי קובץ
    ספירת_חלקים: dict[str, int] = {}
    for מטא in תוצאות["metadatas"]:
        שם = מטא["source"]
        ספירת_חלקים[שם] = ספירת_חלקים.get(שם, 0) + 1

    print("\n--- מקורות ב-ChromaDB ---")
    if not ספירת_חלקים:
        print("אין מסמכים שמורים עדיין.")
    else:
        for שם_קובץ, מספר_חלקים in sorted(ספירת_חלקים.items()):
            print(f"  {שם_קובץ}: {מספר_חלקים} חלקים")


def delete_source(source_name: str) -> int:
    """
    מוחק את כל ה-chunks של קובץ ספציפי מ-ChromaDB.
    מחזיר את מספר ה-chunks שנמחקו.
    """
    לקוח = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח.get_or_create_collection(name="pdf_collection")

    # שולף את מזהי כל ה-chunks של הקובץ
    תוצאות = אוסף.get(where={"source": source_name}, include=["metadatas"])
    מזהים = תוצאות["ids"]

    if not מזהים:
        return 0

    # מוחק את כולם בבת אחת
    אוסף.delete(ids=מזהים)
    return len(מזהים)


def hybrid_search(
    question_en: str,
    collection,
    filter_source: str | None = None,
    n_results: int = 20,
    k_rrf: int = 60,
) -> tuple[list[str], list[str]]:
    """
    מחזיר (texts, sources) באמצעות Hybrid Search:
    - BM25 על כל ה-chunks
    - חיפוש סמנטי ב-ChromaDB
    - שילוב 50/50 באמצעות Reciprocal Rank Fusion
    """
    where_filter = {"source": filter_source} if filter_source else None

    # שלף את כל ה-chunks הרלוונטיים
    כל_הביאה = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )
    כל_טקסטים = כל_הביאה["documents"]
    כל_מטא     = כל_הביאה["metadatas"]
    כל_מזהים  = כל_הביאה["ids"]

    if not כל_טקסטים:
        return [], [], [], [], []

    # מיפוי id -> (chunk_text, source, page_number, full_page_content)
    מזהה_לתוכן = {
        כל_מזהים[i]: (
            כל_טקסטים[i],
            כל_מטא[i]["source"],
            כל_מטא[i].get("page_number", כל_מטא[i].get("chunk_index", 0) + 1),
            כל_מטא[i].get("full_page_content", כל_טקסטים[i]),  # fallback: החזר chunk עצמו
        )
        for i in range(len(כל_מזהים))
    }

    # --- BM25 על כל ה-chunks ---
    קורפוס_מטוקנן = [ט.lower().split() for ט in כל_טקסטים]
    bm25 = BM25Okapi(קורפוס_מטוקנן)
    שאילתא_מטוקננת = question_en.lower().split()
    ציוני_bm25 = bm25.get_scores(שאילתא_מטוקננת)

    # מיון BM25 לפי ציון יורד
    סדר_bm25 = sorted(range(len(כל_טקסטים)), key=lambda i: ציוני_bm25[i], reverse=True)

    # --- חיפוש סמנטי ב-ChromaDB: מספר תוצאות דינמי לפי n_results ---
    n_sem = min(n_results, len(כל_טקסטים))
    תוצאות_סם = collection.query(
        query_texts=[question_en],
        n_results=n_sem,
        where=where_filter,
    )
    מזהיים_סם = תוצאות_סם["ids"][0]

    # --- Reciprocal Rank Fusion (50/50) ---
    ציוני_rrf: dict[str, float] = {}

    # תרומת BM25 (50%)
    for דרגה, אי in enumerate(סדר_bm25):
        מזהה = כל_מזהים[אי]
        ציוני_rrf[מזהה] = ציוני_rrf.get(מזהה, 0.0) + 0.5 / (דרגה + k_rrf)

    # תרומת סמנטיקה (50%)
    for דרגה, מזהה in enumerate(מזהיים_סם):
        ציוני_rrf[מזהה] = ציוני_rrf.get(מזהה, 0.0) + 0.5 / (דרגה + k_rrf)

    # מיון סופי ובחירת top n_results
    מזהיים_מוויינים = sorted(ציוני_rrf, key=ציוני_rrf.__getitem__, reverse=True)[:n_results]

    # בונה את רשימות התוצאות
    טקסטים_סופיים  = []
    מקורות_סופיים  = []
    עמודים_סופיים  = []
    פול_סופיים    = []  # full_page_content
    for מזהה in מזהיים_מוויינים:
        if מזהה in מזהה_לתוכן:
            טקסט, מקור, עמוד, פול = מזהה_לתוכן[מזהה]
            טקסטים_סופיים.append(טקסט)
            מקורות_סופיים.append(מקור)
            עמודים_סופיים.append(עמוד)
            פול_סופיים.append(פול)

    ציוני_סופיים = [ציוני_rrf[מיד] for מיד in מזהיים_מוויינים if מיד in מזהה_לתוכן]
    return טקסטים_סופיים, מקורות_סופיים, ציוני_סופיים, עמודים_סופיים, פול_סופיים


def debug_search(question: str, filter_source: str | None = None) -> None:
    """
    מצב DEBUG: מתרגם את השאלה, מריץ hybrid_search,
    ומדפיס את כל ה-chunks שנשלפו עם ציוני RRF שלהם.
    """
    לקוח_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # תרגום מקצועי-הנדסי לאנגלית (זהה לשלב 1 ב-search_and_answer)
    תגובת_תרגום = לקוח_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "Translate the following technical query to professional engineering English. "
                "Use accurate engineering terminology. Examples: "
                "שנאים→Transformers, משאבות→Pumps, צינורות→Pipes/Conduits, מאגר→Reservoir, "
                "תחנת שאיבה→Pumping Station, קוטר→Diameter, לחץ→Pressure, ספיקה→Flow Rate, "
                "מנהרה→Tunnel, סכר→Dam, מפעל מים→Water Treatment Plant, "
                "מחולל→Generator, לוח חשמל→Switchboard, כבל→Cable, עמוד→Pole/Column. "
                "Return ONLY the translated query, no explanation.\n\n"
                f"Question: {question}"
            ),
        }],
    )
    שאלה_באנגלית = תגובת_תרגום.content[0].text.strip()

    # חיפוש hybrid
    לקוח_chroma = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח_chroma.get_or_create_collection(name="pdf_collection")
    טקסטים, מקורות, ציונים, עמודים, _ = hybrid_search(
        question_en=שאלה_באנגלית,
        collection=אוסף,
        filter_source=filter_source,
        n_results=20,
    )

    # הדפסת תוצאות debug
    print(f"\n{'='*60}")
    print(f"🔍 DEBUG MODE")
    print(f"שאלה מקורית : {question}")
    print(f"תרגום לאנגלית: {שאלה_באנגלית}")
    print(f"chunks שנשלפו: {len(טקסטים)}")
    print(f"{'='*60}\n")

    for i, (טקסט, מקור, ציון, עמוד) in enumerate(zip(טקסטים, מקורות, ציונים, עמודים), start=1):
        תצוגה = טקסט[:200].replace("\n", " ")
        print(f"[{i:02d}] RRF={ציון:.6f} | עמוד {עמוד} | {מקור}")
        print(f"      {תצוגה}{'...' if len(טקסט) > 200 else ''}")
        print()

    print(f"{'='*60}\n")


def search_and_answer(
    question: str,
    history: list[tuple[str, str]] | None = None,
    filter_source: str | None = None,
) -> str:
    """
    מתרגם את השאלה לאנגלית, מחפש ב-ChromaDB את 15 החלקים הרלוונטיים,
    ושולח אותם יחד עם השאלה המקורית והיסטוריית השיחה ל-Anthropic API.
    history: רשימה של (שאלה, תשובה) מהסבבים הקודמים.
    filter_source: אם מועבר, מחפש רק בתוך הקובץ הזה.
    """
    # 🥚 Easter egg – תשובה קשיחה לשאלה הכי חשובה בפרויקט
    if "מנהל" in question and ("הכי טוב" in question or "הטוב ביותר" in question):
        return "ברור שמאיר אזרד 🏆"

    לקוח_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # --- שלב 1: תרגום מקצועי-הנדסי לאנגלית לשיפור החיפוש הסמנטי ---
    תגובת_תרגום = לקוח_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "Translate the following technical query to professional engineering English. "
                "Use accurate engineering terminology. Examples: "
                "שנאים→Transformers, משאבות→Pumps, צינורות→Pipes/Conduits, מאגר→Reservoir, "
                "תחנת שאיבה→Pumping Station, קוטר→Diameter, לחץ→Pressure, ספיקה→Flow Rate, "
                "מנהרה→Tunnel, סכר→Dam, מפעל מים→Water Treatment Plant, "
                "מחולל→Generator, לוח חשמל→Switchboard, כבל→Cable, עמוד→Pole/Column. "
                "Return ONLY the translated query, no explanation.\n\n"
                f"Question: {question}"
            ),
        }],
    )
    שאלה_באנגלית = תגובת_תרגום.content[0].text.strip()

    # --- שלב 1.5: הרחבת שאילתה – 2 גרסאות אנגלית נוספות ---
    תגובת_הרחבה = לקוח_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "Generate 2 alternative English search queries for the following question. "
                "Query 1: use abbreviated technical units (e.g. 'mio m3', 'MCM', 'million m3', 'Mm3'). "
                "Query 2: use expanded descriptive terms (e.g. 'million cubic meters', 'storage capacity', 'total volume'). "
                "Return ONLY the 2 queries, one per line, no numbering or explanation.\n\n"
                f"Original query: {שאלה_באנגלית}"
            ),
        }],
    )
    גרסאות_נוספות = [
        ש.strip()
        for ש in תגובת_הרחבה.content[0].text.strip().splitlines()
        if ש.strip()
    ][:2]  # מקסימום 2 גרסאות נוספות
    כל_שאילתות = [שאלה_באנגלית] + גרסאות_נוספות

    # --- שלב 2: Hybrid Search על כל גרסאות השאילתה + מיזוג תוצאות ---
    לקוח_chroma = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח_chroma.get_or_create_collection(name="pdf_collection")

    # מיזוג לפי מפתח (source, page) – שומר ציון RRF גבוה ביותר מכל הגרסאות
    מיטב_לפי_עמוד: dict[tuple, dict] = {}

    for שאילתה_רחבה in כל_שאילתות:
        _, מקורות_q, ציונים_q, עמודים_q, פולים_q = hybrid_search(
            question_en=שאילתה_רחבה,
            collection=אוסף,
            filter_source=filter_source,
            n_results=50,  # top_k מוגדל ל-50 לרשת רחבה יותר
        )
        for מקור, ציון, עמוד, פול in zip(מקורות_q, ציונים_q, עמודים_q, פולים_q):
            מפתח = (מקור, עמוד)
            if מפתח not in מיטב_לפי_עמוד or ציון > מיטב_לפי_עמוד[מפתח]["ציון"]:
                מיטב_לפי_עמוד[מפתח] = {
                    "ציון": ציון,
                    "full": פול,
                    "source": מקור,
                    "page": עמוד,
                }

    # --- שלב 2.5: Reranking עם Cohere (אם זמין) או מיון RRF רגיל ---
    MAX_PAGES = 10       # עמודים שמגיעים ל-Claude בסופו של דבר
    RERANK_WINDOW = 100  # עמודים מקסימליים שנשלחים ל-Cohere לדירוג
    ממוין_ראשוני = sorted(מיטב_לפי_עמוד.values(), key=lambda x: x["ציון"], reverse=True)

    _cohere_key = os.environ.get("COHERE_API_KEY")
    print(f"[CHECK] Cohere API Key detected: {'YES' if _cohere_key else 'NO'}")

    if _COHERE_AVAILABLE and _cohere_key:
        print("[CHECK] Reranker Status: ACTIVE (Using Cohere)")
        # Reranking: שולח עד RERANK_WINDOW עמודים ל-Cohere, מקבל חזרה MAX_PAGES
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
        for i, item in enumerate(ממוין, 1):
            print(f"  [{i:02d}] Cohere={item['cohere_score']:.4f} | עמוד {item['page']} | {item['source']}")
    else:
        print("[CHECK] Reranker Status: INACTIVE (Fallback to RRF)")
        # fallback: מיון לפי ציון RRF ללא Cohere
        ממוין = ממוין_ראשוני[:MAX_PAGES]
        for i, item in enumerate(ממוין, 1):
            print(f"  [{i:02d}] RRF={item['ציון']:.6f} | עמוד {item['page']} | {item['source']}")

    קטעי_הקשר = [
        f"[מקור: {item['source']} | עמוד {item['page']}]\n{item['full']}"
        for item in ממוין
    ]

    הקשר = "\n\n---\n\n".join(קטעי_הקשר)

    # --- שלב 3: בניית רשימת ההודעות כולל היסטוריית השיחה ---
    system_prompt = (
        "You are a strict document assistant.\n"
        "ABSOLUTE RULES:\n"
        "1. If information is NOT in the provided context - say ONLY: 'המידע לא נמצא בקטעים שנשלפו'\n"
        "2. NEVER guess, estimate, or use prior knowledge\n"
        "3. NEVER apologize or explain - just state clearly what was found or not found\n"
        "4. Always cite the source filename AND page number (עמוד) for every piece of information\n"
        "5. Volume data may appear as full numbers (e.g. 1,164,000 m3) OR in millions "
        "(e.g. 1.18 mio m3 = 1,180,000 m3). Treat them as equivalent when cross-referencing data.\n"
        "6. CONFLICT DETECTION: Scan ALL retrieved chunks. If different numerical values or facts "
        "appear for the same data point — whether across documents or within one document — "
        "flag it explicitly: '⚠️ שים לב: נמצא מידע סותר בין המקורות'\n"
        "7. SOURCE COMPARISON: Never assume the first document is the correct one. "
        "Present ALL versions found in the context and cite each with its source filename and page number.\n\n"
        "ענה בשפה שבה נשאלת השאלה המקורית (עברית או אנגלית).\n"
        f"הקשר מהמסמכים:\n{הקשר}"
    )

    # בונה את רשימת ההודעות: היסטוריה + שאלה נוכחית
    הודעות = []
    for שאלה_קודמת, תשובה_קודמת in (history or []):
        הודעות.append({"role": "user",      "content": שאלה_קודמת})
        הודעות.append({"role": "assistant", "content": תשובה_קודמת})

    # מוסיף את השאלה הנוכחית
    הודעות.append({
        "role": "user",
        "content": f"שאלה מקורית: {question}\n(תורגמה לחיפוש: {שאלה_באנגלית})",
    })

    תגובה = לקוח_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=הודעות,
    )

    return תגובה.content[0].text


def summarize_file(source_name: str) -> str:
    """
    שולף את כל ה-chunks של קובץ ספציפי מ-ChromaDB
    ושולח אותם ל-Claude לסיכום מקיף.
    """
    לקוח_chroma = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח_chroma.get_or_create_collection(name="pdf_collection")

    # שולף את כל הרשומות ששייכות לקובץ המבוקש
    תוצאות = אוסף.get(
        where={"source": source_name},
        include=["documents", "metadatas"]
    )

    חלקים = תוצאות["documents"]

    if not חלקים:
        return f"לא נמצאו נתונים לקובץ: {source_name}"

    # מחבר את כל החלקים לטקסט אחד
    טקסט_מלא = "\n\n".join(חלקים)

    פרומפט = f"""סכם את המסמך הבא בצורה מקיפת ומסודרת.
ציין את הנקודות העיקריות, נושאי המסמך, וכל מידע חשוב אחר.
ענה בעברית.

שם המסמך: {source_name}

תוכן המסמך:
{טקסט_מלא}"""

    # שולח ל-Claude את כל התוכן
    לקוח_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    תגובה = לקוח_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": פרומפט}],
    )

    return תגובה.content[0].text


def count_standards(source_name: str) -> str:
    """
    קורא את ה-PDF ישירות (לא דרך chunks של ChromaDB) כדי למנוע
    פיצול של רשומות תקנים באמצע שורה.
    מעבד 80 שורות בכל קריאת API ומונה את כל התקנים.
    """
    # קורא את ה-PDF מהתיקייה ישירות
    נתיב_pdf = os.path.join("pdfs", source_name)
    if not os.path.exists(נתיב_pdf):
        return f"קובץ לא נמצא: {נתיב_pdf}"

    טקסט_גולמי = load_pdf(נתיב_pdf)

    # מפצל לשורות ומסיר שורות ריקות – כל תקן נשאר שלם בשורתו
    שורות = [ש.strip() for ש in טקסט_גולמי.splitlines() if ש.strip()]

    לקוח_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    גודל_batch = 40  # שורות לכל קריאת API – קטן כדי שהפלט לא יקוצץ
    כל_התקנים: set[str] = set()

    for התחלה in range(0, len(שורות), גודל_batch):
        קבוצה = שורות[התחלה: התחלה + גודל_batch]
        טקסט_קבוצה = "\n".join(קבוצה)

        # מבקש רק מזהים, פלט מינימלי למניעת קיצוץ
        פרומפט_batch = f"""מתוך הטקסט הבא, חלץ את כל התקנים שמופיעים.
החזר שורה אחת לכל תקן, בפורמט המקורי כפי שמופיע בטקסט.
אל תוסיף הסברים, כותרות או מספור – רק את הרשימה.

טקסט:
{טקסט_קבוצה}"""

        תגובה = לקוח_anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": פרומפט_batch}],
        )

        # מוסיף לset – כפילויות מוסרות אוטומטית
        for שורה in תגובה.content[0].text.strip().splitlines():
            שורה = שורה.strip()
            if שורה:
                כל_התקנים.add(שורה)

        print(f"  עיבד שורות {התחלה + 1}–{min(התחלה + גודל_batch, len(שורות))} / {len(שורות)}")

    # פלט מסודר
    רשימה_ממוינת = sorted(כל_התקנים)
    פלט = "\n".join(f"{i}. {תקן}" for i, תקן in enumerate(רשימה_ממוינת, start=1))
    return f"{פלט}\n\nסך הכל: {len(רשימה_ממוינת)} תקנים"


def main():
    """
    פונקציה ראשית:
    1. טוענת את כל ה-PDF מתיקיית 'pdfs'
    2. מפצלת כל טקסט לחלקים
    3. מדפיסה סטטיסטיקות
    """
    תיקיית_pdf = "pdfs"

    # בדיקה אילו קבצים כבר קיימים ב-ChromaDB
    מקורות_קיימים = get_existing_sources()

    # טעינת כל קבצי ה-PDF מהתיקייה
    כל_קבצי_pdf = load_multiple_pdfs(תיקיית_pdf)

    if not כל_קבצי_pdf:
        print("לא נמצאו קבצי PDF בתיקייה.")
        return

    # מסנן רק קבצים חדשים שעוד לא נוספו ל-ChromaDB
    קבצים_חדשים = [ק for ק in כל_קבצי_pdf if ק["source"] not in מקורות_קיימים]
    קבצים_קיימים = [ק for ק in כל_קבצי_pdf if ק["source"] in מקורות_קיימים]

    # מדפיס סטטוס קבצים קיימים
    for קובץ in קבצים_קיימים:
        print(f"דולג (כבר קיים): {קובץ['source']}")

    סה_כ_chunks_חדשים = 0

    # מעבד כל קובץ חדש בנפרד עמוד-עמוד לחיסכון בזיכרון
    for קובץ in קבצים_חדשים:
        נתיב = os.path.join(תיקיית_pdf, קובץ["source"])
        עמודים = count_pdf_pages(נתיב)
        print(f"\nמעבד: {קובץ['source']} ({עמודים} עמודים)")

        def הדפסת_התקדמות(עמוד, סה_כ):
            if עמוד % 20 == 0 or עמוד == סה_כ:
                print(f"  עמוד {עמוד}/{סה_כ}", end="\r", flush=True)

        chunks = process_large_pdf(נתיב, קובץ["source"], progress_callback=הדפסת_התקדמות)
        סה_כ_chunks_חדשים += chunks
        print(f"  ✓ {קובץ['source']}: {chunks} חלקים נשמרו")

    # הדפסת סיכום
    print(f"\n--- סיכום ---")
    print(f"קבצים חדשים שנטענו: {len(קבצים_חדשים)}")
    print(f"קבצים שכבר היו קיימים: {len(קבצים_קיימים)}")
    print(f"חלקים חדשים שנוספו: {סה_כ_chunks_חדשים}")

    # הצגת מקורות קיימים לפני השאלות
    list_sources()

    # לולאת שאלות ותשובות – תומך עוד בפקודת סיכום וספירה
    print("\n=== מוכן! דוגמאות: שאל רגיל | סכם: | ספור: | יציאה ===")
    while True:
        שאלה = input("\nשאלה: ").strip()

        # בודק אם המשתמש רוצה לצאת
        if שאלה.lower() in ("יציאה", "exit"):
            print("להתראות!")
            break

        if not שאלה:
            continue

        # בוחר פקודה לפי הפרפיקס שהמשתמש כתב
        if שאלה.lower().startswith("debug:"):
            שאלה_בדיקה = שאלה.split(":", 1)[1].strip()
            if not שאלה_בדיקה:
                print("נא הקלד שאלה אחרי debug:")
            else:
                debug_search(שאלה_בדיקה)

        elif שאלה.startswith("סכם:") or שאלה.startswith("ספור:"):
            פקודה = "סכם" if שאלה.startswith("סכם:") else "ספור"

            # מציג את הקבצים הזמינים
            מקורות_קיימים_רשימה = sorted(get_existing_sources())
            if not מקורות_קיימים_רשימה:
                print("אין מסמכים זמינים.")
                continue
            print("קבצים זמינים:")
            for י, שם in enumerate(מקורות_קיימים_רשימה, start=1):
                print(f"  {י}. {שם}")
            בחירה = input("בחר מספר קובץ: ").strip()

            # מאפשר בחירה לפי מספר או שם
            if בחירה.isdigit() and 1 <= int(בחירה) <= len(מקורות_קיימים_רשימה):
                שם_קובץ = מקורות_קיימים_רשימה[int(בחירה) - 1]
            elif בחירה in מקורות_קיימים_רשימה:
                שם_קובץ = בחירה
            else:
                print("בחירה לא תקינה.")
                continue

            # קורא לפנקציה המתאימה לפקודה
            if פקודה == "סכם":
                print(f"\nמסכם את: {שם_קובץ}...")
                תשובה = summarize_file(שם_קובץ)
                print(f"\nסיכום:\n{תשובה}")
            else:
                print(f"\nסופר תקנים ב: {שם_קובץ}...")
                תשובה = count_standards(שם_קובץ)
                print(f"\nתוצאה:\n{תשובה}")

        else:
            # מצב רגיל – חיפוש סמנטי ותשובה
            תשובה = search_and_answer(שאלה)
            print(f"\nתשובה:\n{תשובה}")


if __name__ == "__main__":
    main()
