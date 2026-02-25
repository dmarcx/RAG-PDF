import os
import streamlit as st
from dotenv import load_dotenv

# טוען משתני סביבה מ-.env
load_dotenv()

# מייבא את כל הפונקציות הקיימות מ-rag.py
from rag import (
    load_pdf,
    split_text,
    save_to_chromadb,
    get_existing_sources,
    list_sources,
    search_and_answer,
    summarize_file,
    count_pdf_pages,
    process_large_pdf,
    delete_source,
)

# ========================
# הגדרות בסיסיות של הדף
# ========================
st.set_page_config(
    page_title="RAG-PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================
# כותרת ראשית
# ========================
st.markdown("# MANARA Project")
st.title("📄 Ask questions about Manara Project BOD documents")
st.markdown("---")

# ========================
# סרגל צד – מקורות קיימים + העלאת קבצים
# ========================
with st.sidebar:
    st.header("📂 מסמכים טעונים")

    # שולף ומציג את הקבצים הקיימים ב-ChromaDB
    מקורות = sorted(get_existing_sources())
    if מקורות:
        for שם in מקורות:
            עמ1, עמ2 = st.columns([0.82, 0.18])
            עמ1.markdown(f"✅ {שם}")
            # כפתור מחיקה קטן ליד כל קובץ
            if עמ2.button("🗑", key=f"del_{שם}", help=f"מחק את {שם}"):
                st.session_state["pending_delete"] = שם

        # אישור מחיקה – מוצג רק כשנלחץ כפתור מחיקה
        if "pending_delete" in st.session_state:
            שם_למחיקה = st.session_state["pending_delete"]
            st.warning(f"למחוק את **{שם_למחיקה}** מה-DB?")
            אישור, ביטול = st.columns(2)
            if אישור.button("✅ כן, מחק", key="confirm_delete", use_container_width=True):
                נמחקו = delete_source(שם_למחיקה)
                del st.session_state["pending_delete"]
                st.success(f"נמחקו {נמחקו} chunks של {שם_למחיקה}")
                st.rerun()
            if ביטול.button("❌ ביטול", key="cancel_delete", use_container_width=True):
                del st.session_state["pending_delete"]
                st.rerun()
    else:
        st.info("אין מסמכים טעונים עדיין.")


    st.markdown("---")

    # הגנת סיסמה – גישת ניהול בלבד
    סיסמת_ניהול = st.text_input(
        "", type="password", placeholder="Admin password...",
        key="admin_pwd", label_visibility="collapsed",
    )
    מורשה_ניהול = סיסמת_ניהול == "UPLOAD"

    if מורשה_ניהול:
        # ========================
        # העלאת קבצי PDF חדשים
        # ========================
        st.header("⬆️ העלה מסמך חדש")
        קבצים_שהועלו = st.file_uploader(
            "בחר קובץ PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if קבצים_שהועלו:
            if st.button("📥 טען לתוך המערכת", use_container_width=True):
                # מוודא שתיקיית pdfs קיימת
                os.makedirs("pdfs", exist_ok=True)
                מקורות_קיימים = get_existing_sources()
                נוספו = 0

                for קובץ in קבצים_שהועלו:
                    if קובץ.name in מקורות_קיימים:
                        st.warning(f"כבר קיים: {קובץ.name}")
                        continue

                    # שמירה לדיסק בתיקיית pdfs
                    os.makedirs("pdfs", exist_ok=True)
                    נתיב = os.path.join("pdfs", קובץ.name)
                    with open(נתיב, "wb") as f:
                        f.write(קובץ.getbuffer())

                    # מציג שורת סטטוס + progress bar לקבצים גדולים
                    st.markdown(f"**מעבד:** {קובץ.name}")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(עמוד, סה_כ, _bar=progress_bar, _txt=status_text):
                        _bar.progress(עמוד / סה_כ)
                        _txt.caption(f"עמוד {עמוד} / {סה_כ}")

                    chunks = process_large_pdf(נתיב, קובץ.name, progress_callback=progress_callback)
                    progress_bar.progress(1.0)
                    status_text.caption(f"✅ נשמרו {chunks} חלקים")
                    נוספו += 1

                if נוספו > 0:
                    st.success(f"נוספו {נוספו} קובץ/קבצים בהצלחה!")
                    st.rerun()

        st.markdown("---")

        # ========================
        # סריקת תיקיית pdfs קיימת
        # ========================
        st.header("🔍 סרוק תיקיית pdfs")
        st.caption("מאנדקס קבצים שהועתקו ידנית לתיקייה")

        if st.button("🔄 סרוק ואנדקס קבצים חדשים", use_container_width=True):
            תיקיית_pdf = "pdfs"
            if not os.path.isdir(תיקיית_pdf):
                st.error("תיקיית pdfs לא קיימת.")
            else:
                # מוצא קבצים בתיקייה שעוד לא ב-ChromaDB
                מקורות_קיימים = get_existing_sources()
                כל_קבצי_pdf = [
                    ש for ש in os.listdir(תיקיית_pdf)
                    if ש.lower().endswith(".pdf")
                ]
                קבצים_חדשים = [ש for ש in כל_קבצי_pdf if ש not in מקורות_קיימים]

                if not קבצים_חדשים:
                    st.info("כל הקבצים בתיקייה כבר מאונדקסים.")
                else:
                    st.info(f"נמצאו {len(קבצים_חדשים)} קבצים חדשים לאינדוקס.")
                    נוספו = 0
                    for שם_קובץ in קבצים_חדשים:
                        נתיב = os.path.join(תיקיית_pdf, שם_קובץ)
                        st.markdown(f"**מעבד:** {שם_קובץ}")
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(עמוד, סה_כ, _bar=progress_bar, _txt=status_text):
                            _bar.progress(עמוד / סה_כ)
                            _txt.caption(f"עמוד {עמוד} / {סה_כ}")

                        chunks = process_large_pdf(נתיב, שם_קובץ, progress_callback=progress_callback)
                        progress_bar.progress(1.0)
                        status_text.caption(f"✅ {chunks} חלקים")
                        נוספו += 1

                    st.success(f"אונדקסו {נוספו} קבצים בהצלחה!")
                    st.rerun()

    else:
        st.caption("🔒 Administrative access required for indexing")

# ========================
# אזור ראשי – שאלות ותשובות
# ========================

# בחירת מצב פעולה
st.subheader("🔧 בחר מצב")
מצב = st.radio(
    "מצב פעולה:",
    options=["❓ שאלה חופשית", "📋 סכם מסמך"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("---")

# ========================
# מצב: שאלה חופשית
# ========================
if מצב == "❓ שאלה חופשית":
    st.subheader("❓ שאלה חופשית")

    # אתחול היסטוריית שיחה ב-session_state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # רשימה של (שאלה, תשובה)

    # סינון לפי מסמך ספציפי + כפתור ניקוי היסטוריה
    col_filter, col_clear = st.columns([3, 1])
    with col_filter:
        אפשרויות_סינון = ["🌐 כל המסמכים"] + מקורות
        בחירת_סינון = st.selectbox(
            "חפש בתוך:",
            אפשרויות_סינון,
            key="source_filter",
            label_visibility="collapsed",
        )
    with col_clear:
        if st.session_state["chat_history"]:
            if st.button("🗑️ נקה", key="clear_history", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()

    # ממיר את הבחירה לפרמטר סינון (None = כל המסמכים)
    סינון_פעיל = None if בחירת_סינון.startswith("🌐") else בחירת_סינון

    if סינון_פעיל:
        st.caption(f"🔍 מחפש רק ב: **{סינון_פעיל}**")

    # הצגת ההיסטוריה כבועות שיחה
    for שאלה_קודמת, תשובה_קודמת in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(שאלה_קודמת)
        with st.chat_message("assistant"):
            st.markdown(תשובה_קודמת)

    # תיבת שאלה חדשה
    שאלה = st.chat_input("שאל שאלה (עברית או אנגלית)...")

    if שאלה:
        if not get_existing_sources():
            st.error("אין מסמכים טעונים. העלה PDF תחילה.")
        else:
            # מציג את שאלת המשתמש מיד
            with st.chat_message("user"):
                st.markdown(שאלה)

            # שולח לClaude עם כל ההיסטוריה + סינון קובץ
            with st.chat_message("assistant"):
                with st.spinner("מחפש תשובה..."):
                    תשובה = search_and_answer(
                        שאלה,
                        history=st.session_state["chat_history"],
                        filter_source=סינון_פעיל,
                    )
                st.markdown(תשובה)

            # שומר ב-session_state
            st.session_state["chat_history"].append((שאלה, תשובה))

# ========================
# מצב: סיכום מסמך
# ========================
elif מצב == "📋 סכם מסמך":
    st.subheader("📋 סכם מסמך")

    if not מקורות:
        st.error("אין מסמכים טעונים. העלה PDF תחילה.")
    else:
        קובץ_נבחר = st.selectbox("בחר מסמך לסיכום:", מקורות)

        if st.button("✍️ סכם", type="primary", use_container_width=False):
            with st.spinner(f"מסכם את {קובץ_נבחר}..."):
                סיכום = summarize_file(קובץ_נבחר)
            st.markdown("### 📄 סיכום")
            st.markdown(סיכום)

