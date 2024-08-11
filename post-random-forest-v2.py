import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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

# Inisialisasi objek klasifikasi Random Forest
random_forest_classifier = RandomForestClassifier()

# Mengukur waktu untuk fitting model pada data latih
start_fit = time.time()
random_forest_classifier.fit(train_tfidf, train_data["Sentiment_Label"])
end_fit = time.time()

# Mengukur waktu untuk prediksi pada data uji
start_predict = time.time()
test_predictions = random_forest_classifier.predict(test_tfidf)
end_predict = time.time()

# Menghitung durasi proses fitting dan prediksi
fit_duration = end_fit - start_fit
predict_duration = end_predict - start_predict

# Fitting model pada data latih
random_forest_classifier.fit(train_tfidf, train_data["Sentiment_Label"])

# Prediksi pada data uji
test_predictions_rf = random_forest_classifier.predict(test_tfidf)

# Evaluasi model Random Forest
accuracy_rf = accuracy_score(test_data["Sentiment_Label"], test_predictions_rf)
precision_rf = precision_score(
    test_data["Sentiment_Label"], test_predictions_rf, average=None, zero_division=1
)
recall_rf = recall_score(
    test_data["Sentiment_Label"], test_predictions_rf, average=None
)
f1_rf = f1_score(test_data["Sentiment_Label"], test_predictions_rf, average=None)
confusion_rf = confusion_matrix(test_data["Sentiment_Label"], test_predictions_rf)

print("Accuracy Random Forest:", accuracy_rf)
print("Precision Random Forest:", precision_rf)
print("Recall Random Forest:", recall_rf)
print("F1-score Random Forest:", f1_rf)

print(f"Durasi fitting: {fit_duration:.4f} detik")
print(f"Durasi prediksi: {predict_duration:.4f} detik")

# Mencetak Confusion Matrix untuk Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_rf,
    annot=True,
    fmt="d",
    cmap="Oranges",
    cbar=False,
    xticklabels=random_forest_classifier.classes_,
    yticklabels=random_forest_classifier.classes_,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Random Forest")
plt.show()
