import os
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

    # הדפסת סטטיסטיקות
    print(f"\n--- סיכום ---")
    print(f"קבצים שנטענו: {len(קבצים)}")
    print(f"סך הכל חלקים (chunks): {len(כל_החלקים)}")


if __name__ == "__main__":
    main()
