import sqlite3

class VeritabaniYardimcisi:
    def __init__(self, db_path="veritabani.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.veritabani_olustur()

    def veritabani_olustur(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS kullanicilar (
                kullanici_adi TEXT PRIMARY KEY,
                parola TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS degerlendirmeler (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kullanici_adi TEXT,
                soru TEXT,
                cevap TEXT,
                durum TEXT
            )
        """)
        self.conn.commit()

    def kayit_ol(self, kullanici, parola):
        try:
            self.cursor.execute("INSERT INTO kullanicilar (kullanici_adi, parola) VALUES (?, ?)", (kullanici, parola))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def giris_yap(self, kullanici, parola):
        self.cursor.execute("SELECT * FROM kullanicilar WHERE kullanici_adi=? AND parola=?", (kullanici, parola))
        return self.cursor.fetchone() is not None

    def degerlendirme_ekle(self, kullanici, soru, cevap, durum):
        self.cursor.execute(
            "INSERT INTO degerlendirmeler (kullanici_adi, soru, cevap, durum) VALUES (?, ?, ?, ?)",
            (kullanici, soru, cevap, durum)
        )
        self.conn.commit()

    def tum_degerlendirmeleri_getir(self):
        self.cursor.execute("SELECT id, kullanici_adi, soru, cevap, durum FROM degerlendirmeler ORDER BY id DESC")
        return self.cursor.fetchall()

    def degerlendirme_sil(self, id, kullanici):
        self.cursor.execute("DELETE FROM degerlendirmeler WHERE id = ? AND kullanici_adi = ?", (id, kullanici))
        self.conn.commit()