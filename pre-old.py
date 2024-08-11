import pandas as pd
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
df = pd.read_csv("dataset.csv")
print("File CSV berhasil dibaca.")

# Proses Labeling
print("Menentukan sentimen untuk teks ulasan...")


def label_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


df["sentiment_polarity"] = df["review"].apply(lambda x: TextBlob(x).sentiment.polarity)
print("Sentimen berhasil dibuat.")

# Test
# reviews = [
#     "The best movie I ever saw and I watched hundreds off them from Godfather, Pulp fiction, The green mile, The pianist, Schindler's list, Rear window, The shining, Alien etc. What can you ask more: space exploration, apocalyptic earth, emotions, tears, time manipulation, amazing visuals, maybe the best music score, touching acting, everything you want. SHAME on all the oscar"
# ]

# Proses Case-Folding
print("Melakukan case-folding...")
cleaned_reviews_alpha = [
    " ".join(TextBlob(review.lower()).words) for review in df["review"]
]
print("Case-folding selesai.")

# Membersihkan dari simbol & angka, menghapus whitespaces, dan mengubah apostrophe/short word ke bentuk asalnya
print(
    "Membersihkan dari simbol & angka, menghapus whitespaces, dan mengubah apostrophe/short word..."
)
cleaned_reviews_beta = []
for review in cleaned_reviews_alpha:
    cleaned_tokens = []
    for token in TextBlob(review).words:
        # Menghapus simbol dan angka
        clean_token = "".join(
            e for e in token if e.isalpha() or e in ["'", "-"]
        )  # Hanya mempertahankan huruf, apostrophe, dan dash
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
            # Menggabungkan kata yang telah dibersihkan dari simbol dengan kamus koreksi
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
stop_words = set(stopwords.words("english"))
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
        "Processed_Review": [review for review in cleaned_reviews_epsilon],
        "Sentiment_Polarity": df["sentiment_polarity"],
        "Sentiment_Label": df["sentiment_polarity"].apply(label_sentiment),
    }
)

# Simpan DataFrame ke dalam file CSV
df_final.to_csv("hasil_preprocessing.csv", index=False)
print("Data telah disimpan dalam bentuk CSV.")
