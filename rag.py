import os
import anthropic
import chromadb
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()  # טוען את משתני הסביבה מקובץ .env


def load_pdf(file_path: str) -> str:
    """קורא קובץ PDF יחיד ומחזיר את כל הטקסט שלו כמחרוזת."""
    doc = fitz.open(file_path)
    טקסט_מלא = []

    # עובר על כל עמוד בקובץ ומחלץ טקסט
    for עמוד in doc:
        טקסט_מלא.append(עמוד.get_text())

    doc.close()
    return "\n".join(טקסט_מלא)


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


def split_text(text: str, source_name: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
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


def search_and_answer(question: str) -> str:
    """
    מחפש ב-ChromaDB את 3 החלקים הרלוונטיים ביותר לשאלה,
    ושולח אותם יחד עם השאלה ל-Anthropic API לקבלת תשובה.
    """
    # פותח את מסד הנתונים הוקטורי הקיים
    לקוח_chroma = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח_chroma.get_or_create_collection(name="pdf_collection")

    # מחפש את 3 החלקים הדומים ביותר לשאלה
    תוצאות = אוסף.query(query_texts=[question], n_results=6)
    חלקים_רלוונטיים = תוצאות["documents"][0]  # רשימת טקסטים
    מקורות = [m["source"] for m in תוצאות["metadatas"][0]]

    # בונה הקשר מהחלקים שנמצאו
    הקשר = "\n\n---\n\n".join(
        f"[מקור: {מקור}]\n{טקסט}"
        for מקור, טקסט in zip(מקורות, חלקים_רלוונטיים)
    )

    # בונה את הפרומפט לשליחה למודל
    פרומפט = f"""אתה עוזר שעונה על שאלות בהתבסס על מסמכי PDF.
ענה בשפה שבה נשאלת השאלה (עברית או אנגלית).
ענה רק על בסיס המידע הנתון. ציין מאיזה קובץ המידע מגיע.
אם התשובה לא נמצאת בהקשר, אמור זאת בפירוש.

הקשר מהמסמכים:
{הקשר}

שאלה: {question}"""

    # קורא ל-Anthropic API עם הפרומפט
    לקוח_anthropic = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    תגובה = לקוח_anthropic.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": פרומפט}],
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

    כל_החלקים_חדשים = []

    # מעבד ושומר רק קבצים חדשים
    for קובץ in קבצים_חדשים:
        חלקים = split_text(קובץ["text"], קובץ["source"])
        כל_החלקים_חדשים.extend(חלקים)

    if כל_החלקים_חדשים:
        print("שומר קבצים חדשים ב-ChromaDB...")
        save_to_chromadb(כל_החלקים_חדשים)

    # הדפסת סיכום
    print(f"\n--- סיכום ---")
    print(f"קבצים חדשים שנטענו: {len(קבצים_חדשים)}")
    print(f"קבצים שכבר היו קיימים: {len(קבצים_קיימים)}")
    print(f"חלקים חדשים שנוספו: {len(כל_החלקים_חדשים)}")

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
        if שאלה.startswith("סכם:") or שאלה.startswith("ספור:"):
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
