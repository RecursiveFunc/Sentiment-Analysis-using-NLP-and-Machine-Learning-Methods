import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

# print(len(negative_reviews))

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

# Inisialisasi objek k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Mengukur waktu untuk fitting model pada data latih
start_fit = time.time()
knn_classifier.fit(train_tfidf, train_data["Sentiment_Label"])
end_fit = time.time()

# Mengukur waktu untuk prediksi pada data uji
start_predict = time.time()
test_predictions = knn_classifier.predict(test_tfidf)
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
    cmap="Blues",
    cbar=False,
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix KNN")
plt.show()

###############################

# List untuk menyimpan nilai-nilai K dan akurasi yang sesuai
k_values = []
accuracies = []

# Range nilai K yang ingin Anda coba
k_range = range(1, 21)  # Ubah sesuai kebutuhan

# Loop melalui nilai-nilai K
for k in k_range:
    # Inisialisasi k-NN classifier dengan nilai K saat ini
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Fitting model pada data latih
    knn_classifier.fit(train_tfidf, train_data["Sentiment_Label"])

    # Prediksi pada data uji
    test_predictions = knn_classifier.predict(test_tfidf)

    # Hitung dan simpan akurasi
    accuracy = accuracy_score(test_data["Sentiment_Label"], test_predictions)
    k_values.append(k)
    accuracies.append(accuracy)

# Plot grafik akurasi vs. nilai K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker="o", linestyle="-")
plt.title("Accuracy vs. Number of Neighbors (K)")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.xticks(k_range)
plt.grid(True)
plt.show()
