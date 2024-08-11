import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import time

######### Membaca File CSV #########
try:
    print("Membaca file CSV...")
    df = pd.read_csv("hasil_preprocessing_new_v2.csv")
    print("File CSV berhasil dibaca.")
except FileNotFoundError:
    print("File CSV tidak ditemukan. Pastikan path file CSV sudah benar.")
    exit()
except Exception as e:
    print("Terjadi kesalahan saat membaca file CSV:", e)
    exit()

###############################

# Memisahkan menjadi ulasan positif, negatif, dan netral berdasarkan sentimennya
print("Memisahkan ulasan berdasarkan sentimen...")
positive_reviews = df[df["Sentiment_Label"] == "positive"]
negative_reviews = df[df["Sentiment_Label"] == "negative"]
print("Ulasan berhasil dipisahkan.")

# Menggabungkan hasil pemisahan sentimen menjadi satu DataFrame
all_reviews = pd.concat([positive_reviews, negative_reviews])

# Pembagian data latih dan data uji
test_size = 0.2  # 80:20
random_state = 42  #

# Memisahkan ulasan menjadi data latih dan data uji
train_data, test_data = train_test_split(
    all_reviews, test_size=test_size, random_state=random_state
)

################################

# Inisialisasi objek TF-IDF dengan vektorisasi hanya pada data latih
tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(train_data["Processed_Review"])

# Melakukan transform pada data uji
test_tfidf = tfidf_vectorizer.transform(test_data["Processed_Review"])

################################

# Inisialisasi objek klasifikasi Naive Bayes
naive_bayes_classifier = MultinomialNB()

# Mengukur waktu untuk fitting model pada data latih
start_fit = time.time()
naive_bayes_classifier.fit(train_tfidf, train_data["Sentiment_Label"])
end_fit = time.time()

# Mengukur waktu untuk prediksi pada data uji
start_predict = time.time()
test_predictions = naive_bayes_classifier.predict(test_tfidf)
end_predict = time.time()

# Menghitung durasi proses fitting dan prediksi
fit_duration = end_fit - start_fit
predict_duration = end_predict - start_predict

# Fitting model pada data latih
naive_bayes_classifier.fit(train_tfidf, train_data["Sentiment_Label"])

# Prediksi pada data uji
test_predictions_nb = naive_bayes_classifier.predict(test_tfidf)

# Evaluasi model Naive Bayes
accuracy_nb = accuracy_score(test_data["Sentiment_Label"], test_predictions_nb)
precision_nb = precision_score(
    test_data["Sentiment_Label"], test_predictions_nb, average=None, zero_division=1
)
recall_nb = recall_score(
    test_data["Sentiment_Label"], test_predictions_nb, average=None
)
f1_nb = f1_score(test_data["Sentiment_Label"], test_predictions_nb, average=None)
confusion_nb = confusion_matrix(test_data["Sentiment_Label"], test_predictions_nb)

print("Accuracy Naive Bayes:", accuracy_nb)
print("Precision Naive Bayes:", precision_nb)
print("Recall Naive Bayes:", recall_nb)
print("F1-score Naive Bayes:", f1_nb)

print(f"Durasi fitting: {fit_duration:.4f} detik")
print(f"Durasi prediksi: {predict_duration:.4f} detik")

# Mencetak Confusion Matrix untuk Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_nb,
    annot=True,
    fmt="d",
    cmap="Greens",
    cbar=False,
    xticklabels=naive_bayes_classifier.classes_,
    yticklabels=naive_bayes_classifier.classes_,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Naive Bayes")
plt.show()
