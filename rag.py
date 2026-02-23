import os
import anthropic
import chromadb
import fitz  # PyMuPDF


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
    שומר את כל החלקים במסד הנתונים הוקטורי ChromaDB.
    האוסף נקרא 'pdf_collection'.
    כל רשומה מכילה:
      - id: מזהה ייחודי (שם מקור + אינדקס)
      - document: תוכן הטקסט
      - metadata: שם המקור ומספר החלק
    """
    # יוצר לקוח ChromaDB שישמור את הנתונים בתיקייה מקומית
    לקוח = chromadb.PersistentClient(path="chroma_db")

    # מוחק את האוסף הקיים (אם יש) כדי למנוע כפילויות בכל הרצה
    לקוח.delete_collection(name="pdf_collection")

    # יוצר אוסף חדש ונקי
    אוסף = לקוח.create_collection(name="pdf_collection")

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


def search_and_answer(question: str) -> str:
    """
    מחפש ב-ChromaDB את 3 החלקים הרלוונטיים ביותר לשאלה,
    ושולח אותם יחד עם השאלה ל-Anthropic API לקבלת תשובה.
    """
    # פותח את מסד הנתונים הוקטורי הקיים
    לקוח_chroma = chromadb.PersistentClient(path="chroma_db")
    אוסף = לקוח_chroma.get_or_create_collection(name="pdf_collection")

    # מחפש את 3 החלקים הדומים ביותר לשאלה
    תוצאות = אוסף.query(query_texts=[question], n_results=3)
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


def main():
    """
    פונקציה ראשית:
    1. טוענת את כל ה-PDF מתיקיית 'pdfs'
    2. מפצלת כל טקסט לחלקים
    3. מדפיסה סטטיסטיקות
    """
    תיקיית_pdf = "pdfs"

    # טעינת כל קבצי ה-PDF מהתיקייה
    קבצים = load_multiple_pdfs(תיקיית_pdf)

    if not קבצים:
        print("לא נמצאו קבצי PDF בתיקייה.")
        return

    כל_החלקים = []

    # פיצול הטקסט של כל קובץ לחלקים
    for קובץ in קבצים:
        חלקים = split_text(קובץ["text"], קובץ["source"])
        כל_החלקים.extend(חלקים)

    # שמירת כל החלקים ב-ChromaDB
    print("שומר ב-ChromaDB...")
    save_to_chromadb(כל_החלקים)

    # הדפסת סטטיסטיקות
    print(f"\n--- סיכום ---")
    print(f"קבצים שנטענו: {len(קבצים)}")
    print(f"סך הכל חלקים (chunks): {len(כל_החלקים)}")
    print("הנתונים נשמרו בהצלחה ב-ChromaDB (תיקיית chroma_db).")

    # לולאת שאלות ותשובות – ממשיכה עד שהמשתמש מקליד יציאה/exit
    print("\n=== מוכן לשאלות! (כתוב 'יציאה' או 'exit' לסיום) ===")
    while True:
        שאלה = input("\nשאלה: ").strip()

        # בודק אם המשתמש רוצה לצאת
        if שאלה.lower() in ("יציאה", "exit"):
            print("להתראות!")
            break

        if not שאלה:
            continue

        # מחפש ומקבל תשובה מהמודל
        תשובה = search_and_answer(שאלה)
        print(f"\nתשובה:\n{תשובה}")


if __name__ == "__main__":
    main()
