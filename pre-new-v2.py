import pandas as pd
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Proses Collecting Data
print("Membaca file CSV...")
df = pd.read_csv("dataset-50k.csv")
print("File CSV berhasil dibaca.")

stop_words = set(stopwords.words("english"))


def remove_html_tags(text):
    # Menghapus tag HTML <br> dan <br/>
    clean_text = re.sub(r"<br\s*/?>", " ", text)
    # Menghapus semua tag HTML lainnya
    clean_text = re.sub(r"<.*?>", "", clean_text)
    return clean_text


# Membersihkan tag HTML dari kolom "review"
df["cleaned_review"] = df["review"].apply(remove_html_tags)

# Proses Case-Folding
print("Melakukan case-folding...")
cleaned_reviews_alpha = [
    " ".join(TextBlob(review.lower()).words) for review in df["cleaned_review"]
]
print("Case-folding selesai.")

# Membersihkan tag HTML, simbol & angka, menghapus whitespaces, dan mengubah apostrophe/short word ke bentuk asalnya
print(
    "Membersihkan tag HTML, simbol & angka, menghapus whitespaces, dan mengubah apostrophe/short word..."
)
cleaned_reviews_beta = []
for review in cleaned_reviews_alpha:
    cleaned_tokens = []
    for token in review.split():  # Memecah teks ulasan menjadi token
        # Menghapus simbol dan angka
        clean_token = re.sub(r"[^a-zA-Z'-]", "", token)
        if clean_token:
            # Menghapus whitespaces
            clean_token = clean_token.strip()
            # Mengubah apostrophe/short word ke bentuk asalnya
            corrections = {
                "'s": " is",
                "'re": " are",
                "'ll": " will",
                "'ve": " have",
                "'d": " would",
                "n't": " not",
            }
            # Menggabungkan kata yang telah dibersihkan dari tag HTML, simbol, dengan kamus koreksi
            clean_token = corrections.get(clean_token, clean_token)
            cleaned_tokens.append(clean_token)
    cleaned_reviews_beta.append(" ".join(cleaned_tokens))
print("Pembersihan selesai.")


# Proses Tokenisasi
print("Melakukan tokenisasi...")
cleaned_reviews_gamma = [word_tokenize(review) for review in cleaned_reviews_beta]
print("Tokenisasi selesai.")

# Proses Stopwords Removal
print("Menghapus stopwords...")
cleaned_reviews_delta = [
    [word for word in review if word not in stop_words]
    for review in cleaned_reviews_gamma
]
print("Stopwords removal selesai.")

# Proses Lemmatization
print("Melakukan lemmatisasi...")
lemmatizer = WordNetLemmatizer()
cleaned_reviews_epsilon = [
    [lemmatizer.lemmatize(word) for word in review] for review in cleaned_reviews_delta
]
print("Lemmatisasi selesai.")

# Menunjukkan hasil akhir dalam bentuk tabel
df_final = pd.DataFrame(
    {
        "Original_Review": df["review"],
        "Processed_Review": cleaned_reviews_epsilon,
        "Sentiment_Label": df["sentiment"],
    }
)

# Simpan DataFrame ke dalam file CSV
df_final.to_csv("hasil_preprocessing_new_v2.csv", index=False)
print("Data telah disimpan dalam bentuk CSV.")
