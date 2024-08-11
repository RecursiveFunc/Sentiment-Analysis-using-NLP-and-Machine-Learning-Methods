import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import time  # Importing the time module to measure duration

######### Membaca File CSV #########
try:
    print("Membaca file CSV...")
    df = pd.read_csv("hasil_preprocessing_new.csv")
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
positive_reviews = df[df["Sentiment_Polarity"] > 0]
negative_reviews = df[df["Sentiment_Polarity"] < 0]
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

# print(len(test_data))

################################

# Inisialisasi objek TF-IDF dengan vektorisasi hanya pada data latih
tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(train_data["Processed_Review"])

# Melakukan transform pada data uji
test_tfidf = tfidf_vectorizer.transform(test_data["Processed_Review"])

################################

# Inisialisasi objek logistic regression classifier
log_reg_classifier = LogisticRegression(max_iter=1000)

# Mengukur waktu untuk fitting model pada data latih
start_fit = time.time()
log_reg_classifier.fit(train_tfidf, train_data["Sentiment_Label"])
end_fit = time.time()

# Mengukur waktu untuk prediksi pada data uji
start_predict = time.time()
test_predictions = log_reg_classifier.predict(test_tfidf)
end_predict = time.time()

# Menghitung durasi proses fitting dan prediksi
fit_duration = end_fit - start_fit
predict_duration = end_predict - start_predict

################################

# Evaluasi model
accuracy = accuracy_score(test_data["Sentiment_Label"], test_predictions)
precision = precision_score(
    test_data["Sentiment_Label"], test_predictions, average=None
)
recall = recall_score(test_data["Sentiment_Label"], test_predictions, average=None)
f1 = f1_score(test_data["Sentiment_Label"], test_predictions, average=None)
confusion = confusion_matrix(test_data["Sentiment_Label"], test_predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-measure:", f1)

print(f"Durasi fitting: {fit_duration:.4f} detik")
print(f"Durasi prediksi: {predict_duration:.4f} detik")


################################

# Mendefinisikan label
class_labels = ["Negative", "Positive"]

# Mencetak Confussion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion,
    annot=True,
    fmt="d",
    cmap="Purples",
    cbar=False,
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Logistic Regression")
plt.show()

###############################

# Plotting Grafik Akurasi vs. Jumlah Tetangga (K)
# Note: Logistic Regression tidak memiliki parameter 'k', tetapi kita bisa membuat plot akurasi terhadap iterasi jika diperlukan.
