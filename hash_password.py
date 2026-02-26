"""
כלי עזר: יצירת bcrypt hash לסיסמה ו-COOKIE_KEY אקראי
לשימוש ב-.streamlit/secrets.toml

הפעלה:
    python hash_password.py
"""
import getpass
import secrets
import bcrypt


def main():
    print("=" * 55)
    print("  Password Hash Generator for secrets.toml")
    print("=" * 55)

    # יצירת hash לסיסמה
    password = getpass.getpass("\nEnter password to hash: ")
    if not password:
        print("❌ Password cannot be empty.")
        return

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(12)).decode("utf-8")
    print(f"\n✅ Hashed password (copy to secrets.toml):\n{hashed}")

    # יצירת מפתח cookie אקראי
    cookie_key = secrets.token_hex(32)
    print(f"\n✅ Random COOKIE_KEY suggestion (copy to secrets.toml):\n{cookie_key}")

    print("\n" + "=" * 55)
    print("  Copy the values above into .streamlit/secrets.toml")
    print("=" * 55)


if __name__ == "__main__":
    main()
