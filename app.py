import os
import streamlit as st
import streamlit_authenticator as stauth
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
# מילון תרגומים – כל מחרוזות ה-UI
# ========================
TRANSLATIONS = {
    "en": {
        "main_header":       "MANARA Project",
        "main_title":        "📄 Ask questions about Manara Project BOD documents",
        "docs_header":       "📂 Loaded Documents",
        "no_docs":           "No documents loaded yet.",
        "del_help":          "Delete {}",
        "del_confirm":       "Delete **{}** from the DB?",
        "del_yes":           "✅ Yes, Delete",
        "del_cancel":        "❌ Cancel",
        "del_success":       "Deleted {} chunks of {}",
        "admin_placeholder": "Admin password...",
        "admin_locked":      "🔒 Administrative access required for indexing",
        "upload_header":     "⬆️ Upload New Document",
        "upload_label":      "Select PDF file",
        "upload_btn":        "📥 Load into System",
        "already_exists":    "Already exists: {}",
        "processing":        "**Processing:** {}",
        "saved_chunks":      "✅ Saved {} chunks",
        "upload_success":    "Added {} file(s) successfully!",
        "scan_header":       "🔍 Scan pdfs folder",
        "scan_caption":      "Index files copied manually to the folder",
        "scan_btn":          "🔄 Scan & Index New Files",
        "scan_no_folder":    "pdfs folder does not exist.",
        "scan_all_indexed":  "All files in folder are already indexed.",
        "scan_found":        "Found {} new files to index.",
        "scan_chunks":       "✅ {} chunks",
        "scan_success":      "Indexed {} files successfully!",
        "mode_subheader":    "🔧 Select Mode",
        "mode_qa":           "❓ Free Question",
        "mode_summarize":    "📋 Summarize Document",
        "qa_header":         "❓ Free Question",
        "guide_btn":         "📖 User Guide",
        "filter_label":      "Filter documents (leave empty = all):",
        "clear_btn":         "🗑️ Clear",
        "filter_active":     "🔍 Searching only in: **{}**",
        "filter_multi":      "🔍 Searching in {} selected documents",
        "chat_placeholder":  "Ask a question (Hebrew or English)...",
        "no_docs_error":     "No documents loaded. Please upload a PDF first.",
        "searching":         "Searching for answer...",
        "summarize_header":  "📋 Summarize Document",
        "summarize_select":  "Select document to summarize:",
        "summarize_btn":     "✍️ Summarize",
        "summarizing":       "Summarizing {}...",
        "summary_title":     "### 📄 Summary",
        "login_form_name":   "Login",
        "login_username":    "Username",
        "login_password":    "Password",
        "login_btn":         "Login",
        "login_error":       "❌ Incorrect username or password",
        "login_warning":     "Please enter your username and password",
        "logout_btn":        "🚪 Logout",
        "welcome_user":      "👤 {}",
    },
    "he": {
        "main_header":       "פרויקט מנרה",
        "main_title":        "📄 שאל שאלות על מסמכי בסיס נתונים של פרויקט מנרה",
        "docs_header":       "📂 מסמכים טעונים",
        "no_docs":           "אין מסמכים טעונים עדיין.",
        "del_help":          "מחק את {}",
        "del_confirm":       "למחוק את **{}** מה-DB?",
        "del_yes":           "✅ כן, מחק",
        "del_cancel":        "❌ ביטול",
        "del_success":       "נמחקו {} chunks של {}",
        "admin_placeholder": "סיסמת ניהול...",
        "admin_locked":      "🔒 נדרשת גישת מנהל לאינדוקס",
        "upload_header":     "⬆️ העלה מסמך חדש",
        "upload_label":      "בחר קובץ PDF",
        "upload_btn":        "📥 טען לתוך המערכת",
        "already_exists":    "כבר קיים: {}",
        "processing":        "**מעבד:** {}",
        "saved_chunks":      "✅ נשמרו {} חלקים",
        "upload_success":    "נוספו {} קובץ/קבצים בהצלחה!",
        "scan_header":       "🔍 סרוק תיקיית pdfs",
        "scan_caption":      "מאנדקס קבצים שהועתקו ידנית לתיקייה",
        "scan_btn":          "🔄 סרוק ואנדקס קבצים חדשים",
        "scan_no_folder":    "תיקיית pdfs לא קיימת.",
        "scan_all_indexed":  "כל הקבצים בתיקייה כבר מאונדקסים.",
        "scan_found":        "נמצאו {} קבצים חדשים לאינדוקס.",
        "scan_chunks":       "✅ {} חלקים",
        "scan_success":      "אונדקסו {} קבצים בהצלחה!",
        "mode_subheader":    "🔧 בחר מצב",
        "mode_qa":           "❓ שאלה חופשית",
        "mode_summarize":    "📋 סכם מסמך",
        "qa_header":         "❓ שאלה חופשית",
        "guide_btn":         "📖 מדריך משתמש",
        "filter_label":      "סנן מסמכים (ריק = כולם):",
        "clear_btn":         "🗑️ נקה",
        "filter_active":     "🔍 מחפש רק ב: **{}**",
        "filter_multi":      "🔍 מחפש ב-{} מסמכים נבחרים",
        "chat_placeholder":  "שאל שאלה (עברית או אנגלית)...",
        "no_docs_error":     "אין מסמכים טעונים. העלה PDF תחילה.",
        "searching":         "מחפש תשובה...",
        "summarize_header":  "📋 סכם מסמך",
        "summarize_select":  "בחר מסמך לסיכום:",
        "summarize_btn":     "✍️ סכם",
        "summarizing":       "מסכם את {}...",
        "summary_title":     "### 📄 סיכום",
        "login_form_name":   "כניסה",
        "login_username":    "שם משתמש",
        "login_password":    "סיסמה",
        "login_btn":         "כניסה",
        "login_error":       "❌ שם משתמש או סיסמה שגויים",
        "login_warning":     "אנא הכנס שם משתמש וסיסמה",
        "logout_btn":        "🚪 יציאה",
        "welcome_user":      "👤 {}",
    },
}

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
# קוד שפה – נקרא מ-session_state לפני כל רינדור
# session_state["lang"] מאוכלס מהריצה הקודמת (ברירת מחדל: English)
# ========================
קוד_שפה = "he" if st.session_state.get("lang", "English") == "עברית" else "en"


def t(key: str, *args) -> str:
    """מחזיר מחרוזת מתורגמת לפי שפת הממשק הנוכחית."""
    s = TRANSLATIONS[קוד_שפה].get(key, key)
    return s.format(*args) if args else s



# ========================
# CSS לתמיכה ב-RTL בעברית
# ========================
if קוד_שפה == "he":
    st.markdown(
        """
        <style>
        .main .block-container { direction: rtl; text-align: right; }
        .stChatMessage           { direction: rtl; }
        .stChatInput textarea    { direction: rtl; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ========================
# לוגו + כותרת ראשית
# ========================
col_logo = st.columns([1.5, 1, 1.5])[1]  # עמודה מרכזית צרה יותר – לוגו קטן בחצי
with col_logo:
    st.image("SLD LOGO.png", use_container_width=True)

st.markdown(
    f"<h1 style='text-align:center;'>{t('main_header')}</h1>"
    f"<h3 style='text-align:center; color:gray;'>{t('main_title')}</h3>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ========================
# אימות משתמש
# ========================
try:
    # המרת CREDENTIALS מ-AttrDict של Streamlit Secrets ל-dict רגיל
    _raw = st.secrets["CREDENTIALS"]["usernames"]
    _credentials = {
        "usernames": {
            _uname: {
                "email":                 str(_u["email"]),
                "first_name":            str(_u.get("first_name", "")),
                "last_name":             str(_u.get("last_name", "")),
                "password":              str(_u["password"]),
                "roles":                 list(_u.get("roles", ["user"])),
                "failed_login_attempts": int(_u.get("failed_login_attempts", 0)),
                "logged_in":             bool(_u.get("logged_in", False)),
            }
            for _uname, _u in dict(_raw).items()
        }
    }
    _cookie_key = str(st.secrets.get("COOKIE_KEY", "change_me_in_secrets"))
except Exception as _err:
    st.error(f"⚙️ Secrets not configured — edit .streamlit/secrets.toml\n\n`{_err}`")
    st.stop()

_authenticator = stauth.Authenticate(
    _credentials,
    "rag_pdf_session",    # שם הcookie
    _cookie_key,
    cookie_expiry_days=7,
    auto_hash=False,      # סיסמאות כבר hashed ב-secrets.toml
)

_authenticator.login(
    fields={
        "Form name": t("login_form_name"),
        "Username":  t("login_username"),
        "Password":  t("login_password"),
        "Login":     t("login_btn"),
    }
)

if st.session_state.get("authentication_status") is False:
    st.error(t("login_error"))
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning(t("login_warning"))
    st.stop()

# ← מכאן: משתמשים מאומתים בלבד ←

# ========================
# סרגל צד – בורר שפה + מסמכים + העלאה
# ========================
with st.sidebar:

    # שם המשתמש המחובר + כפתור התנתקות
    st.caption(t("welcome_user", st.session_state.get("name", "")))
    _authenticator.logout(
        button_name=t("logout_btn"),
        location="sidebar",
        key="logout_btn",
        use_container_width=True,
    )
    st.markdown("---")

    # כפתור החלפת שפה – מציג את השפה הנגדית; לחיצה מחליפה ומרעננת
    btn_label = "🌐 עב" if st.session_state.get("lang", "English") == "English" else "🌐 EN"
    if st.button(btn_label, key="lang_toggle"):
        st.session_state["lang"] = (
            "עברית" if st.session_state.get("lang", "English") == "English" else "English"
        )
        st.rerun()

    # כפתור מדריך למשתמש – פותח חלון צף עם המדריך בשפה הנוכחית
    with st.popover(t("guide_btn"), use_container_width=True):
        try:
            with open("USER_GUIDE.md", encoding="utf-8") as _f:
                _content = _f.read()
            _parts = _content.split("---\n---\n")
            if קוד_שפה == "he" and len(_parts) > 1:
                st.markdown(_parts[1])
            else:
                st.markdown(_parts[0])
        except FileNotFoundError:
            st.error("USER_GUIDE.md not found.")

    st.markdown("---")
    st.header(t("docs_header"))

    # שולף ומציג את הקבצים הקיימים ב-ChromaDB
    מקורות = sorted(get_existing_sources())
    if מקורות:
        for שם in מקורות:
            עמ1, עמ2 = st.columns([0.82, 0.18])
            עמ1.markdown(f"✅ {שם}")
            # כפתור מחיקה קטן ליד כל קובץ
            if עמ2.button("🗑", key=f"del_{שם}", help=t("del_help", שם)):
                st.session_state["pending_delete"] = שם

        # אישור מחיקה – מוצג רק כשנלחץ כפתור מחיקה
        if "pending_delete" in st.session_state:
            שם_למחיקה = st.session_state["pending_delete"]
            st.warning(t("del_confirm", שם_למחיקה))
            אישור, ביטול = st.columns(2)
            if אישור.button(t("del_yes"), key="confirm_delete", use_container_width=True):
                נמחקו = delete_source(שם_למחיקה)
                del st.session_state["pending_delete"]
                st.success(t("del_success", נמחקו, שם_למחיקה))
                st.rerun()
            if ביטול.button(t("del_cancel"), key="cancel_delete", use_container_width=True):
                del st.session_state["pending_delete"]
                st.rerun()
    else:
        st.info(t("no_docs"))

    st.markdown("---")

    # הגנת סיסמה – גישת ניהול בלבד
    סיסמת_ניהול = st.text_input(
        "", type="password", placeholder=t("admin_placeholder"),
        key="admin_pwd", label_visibility="collapsed",
    )
    מורשה_ניהול = סיסמת_ניהול == "UPLOAD"

    if מורשה_ניהול:
        # ========================
        # העלאת קבצי PDF חדשים
        # ========================
        st.header(t("upload_header"))
        קבצים_שהועלו = st.file_uploader(
            t("upload_label"),
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if קבצים_שהועלו:
            if st.button(t("upload_btn"), use_container_width=True):
                os.makedirs("pdfs", exist_ok=True)
                מקורות_קיימים = get_existing_sources()
                נוספו = 0

                for קובץ in קבצים_שהועלו:
                    if קובץ.name in מקורות_קיימים:
                        st.warning(t("already_exists", קובץ.name))
                        continue

                    # שמירה לדיסק בתיקיית pdfs
                    נתיב = os.path.join("pdfs", קובץ.name)
                    with open(נתיב, "wb") as f:
                        f.write(קובץ.getbuffer())

                    # מציג שורת סטטוס + progress bar לקבצים גדולים
                    st.markdown(t("processing", קובץ.name))
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def progress_callback(עמוד, סה_כ, _bar=progress_bar, _txt=status_text):
                        _bar.progress(עמוד / סה_כ)
                        _txt.caption(f"{עמוד} / {סה_כ}")

                    chunks = process_large_pdf(נתיב, קובץ.name, progress_callback=progress_callback)
                    progress_bar.progress(1.0)
                    status_text.caption(t("saved_chunks", chunks))
                    נוספו += 1

                if נוספו > 0:
                    st.success(t("upload_success", נוספו))
                    st.rerun()

        st.markdown("---")

        # ========================
        # סריקת תיקיית pdfs קיימת
        # ========================
        st.header(t("scan_header"))
        st.caption(t("scan_caption"))

        if st.button(t("scan_btn"), use_container_width=True):
            תיקיית_pdf = "pdfs"
            if not os.path.isdir(תיקיית_pdf):
                st.error(t("scan_no_folder"))
            else:
                # מוצא קבצים בתיקייה שעוד לא ב-ChromaDB
                מקורות_קיימים = get_existing_sources()
                כל_קבצי_pdf = [
                    ש for ש in os.listdir(תיקיית_pdf)
                    if ש.lower().endswith(".pdf")
                ]
                קבצים_חדשים = [ש for ש in כל_קבצי_pdf if ש not in מקורות_קיימים]

                if not קבצים_חדשים:
                    st.info(t("scan_all_indexed"))
                else:
                    st.info(t("scan_found", len(קבצים_חדשים)))
                    נוספו = 0
                    for שם_קובץ in קבצים_חדשים:
                        נתיב = os.path.join(תיקיית_pdf, שם_קובץ)
                        st.markdown(t("processing", שם_קובץ))
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(עמוד, סה_כ, _bar=progress_bar, _txt=status_text):
                            _bar.progress(עמוד / סה_כ)
                            _txt.caption(f"{עמוד} / {סה_כ}")

                        chunks = process_large_pdf(נתיב, שם_קובץ, progress_callback=progress_callback)
                        progress_bar.progress(1.0)
                        status_text.caption(t("scan_chunks", chunks))
                        נוספו += 1

                    st.success(t("scan_success", נוספו))
                    st.rerun()

    else:
        st.caption(t("admin_locked"))

# ========================
# אזור ראשי – שאלות ותשובות
# ========================

# בחירת מצב פעולה
st.subheader(t("mode_subheader"))
מצב = st.radio(
    "mode",
    options=[t("mode_qa"), t("mode_summarize")],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown("---")

# ========================
# מצב: שאלה חופשית
# ========================
if מצב == t("mode_qa"):
    st.subheader(t("qa_header"))

    # אתחול היסטוריית שיחה ב-session_state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # סינון לפי מסמכים נבחרים + כפתור ניקוי היסטוריה
    col_filter, col_clear = st.columns([3, 1])
    with col_filter:
        מסמכים_נבחרים = st.multiselect(
            t("filter_label"),
            options=מקורות,
            default=[],
            key="source_filter",
            label_visibility="collapsed",
            placeholder=t("filter_label"),
        )
    with col_clear:
        if st.session_state["chat_history"]:
            if st.button(t("clear_btn"), key="clear_history", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()

    # ממיר את הבחירה לפרמטר סינון (None = כל המסמכים, רשימה = סינון מרובה)
    if not מסמכים_נבחרים:
        סינון_פעיל = None
    elif len(מסמכים_נבחרים) == 1:
        סינון_פעיל = מסמכים_נבחרים[0]
        st.caption(t("filter_active", סינון_פעיל))
    else:
        סינון_פעיל = מסמכים_נבחרים
        st.caption(t("filter_multi", len(מסמכים_נבחרים)))

    # הצגת ההיסטוריה כבועות שיחה
    for שאלה_קודמת, תשובה_קודמת in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(שאלה_קודמת)
        with st.chat_message("assistant"):
            st.markdown(תשובה_קודמת)

    # תיבת שאלה חדשה
    שאלה = st.chat_input(t("chat_placeholder"))

    if שאלה:
        if not get_existing_sources():
            st.error(t("no_docs_error"))
        else:
            # מציג את שאלת המשתמש מיד
            with st.chat_message("user"):
                st.markdown(שאלה)

            # שולח לClaude עם כל ההיסטוריה + סינון קובץ
            with st.chat_message("assistant"):
                with st.spinner(t("searching")):
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
elif מצב == t("mode_summarize"):
    st.subheader(t("summarize_header"))

    if not מקורות:
        st.error(t("no_docs_error"))
    else:
        קובץ_נבחר = st.selectbox(t("summarize_select"), מקורות)

        if st.button(t("summarize_btn"), type="primary", use_container_width=False):
            with st.spinner(t("summarizing", קובץ_נבחר)):
                סיכום = summarize_file(קובץ_נבחר)
            st.markdown(t("summary_title"))
            st.markdown(סיכום)
