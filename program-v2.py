import tkinter as tk
import pandas as pd
import time
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Function to classify sentiment using Naive Bayes
def classify_sentiment_nb():
    start_time = time.time()  # Start time

    review_text = name.get()
    cleaned_review = preprocess_review(review_text)
    prediction = nb_model.predict(vectorizer.transform([cleaned_review]))[0]
    if prediction == "Positive":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Processing time

    accuracy = nb_model.score(X_test, y_test)  # Accuracy

    greeting = f"The sentiment of your review (Naive Bayes) is {sentiment}, value: {prediction}\n"
    greeting += f"Processing Time: {processing_time:.4f} seconds\n"
    greeting += f"Accuracy: {accuracy:.2f}"

    greeting_label_nb.config(text=greeting)


# Function to classify sentiment using Random Forest
def classify_sentiment_rf():
    start_time = time.time()  # Start time

    review_text = name.get()
    cleaned_review = preprocess_review(review_text)
    prediction = rf_model.predict(vectorizer.transform([cleaned_review]))[0]
    if prediction == "Positive":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Processing time

    accuracy = rf_model.score(X_test, y_test)  # Accuracy

    greeting = f"The sentiment of your review (Random Forest) is {sentiment}, value: {prediction}\n"
    greeting += f"Processing Time: {processing_time:.4f} seconds\n"
    greeting += f"Accuracy: {accuracy:.2f}"

    greeting_label_rf.config(text=greeting)


# Function to classify sentiment using Logistic Regression
def classify_sentiment_lr():
    start_time = time.time()  # Start time

    review_text = name.get()
    cleaned_review = preprocess_review(review_text)
    prediction = lr_model.predict(vectorizer.transform([cleaned_review]))[0]
    if prediction == "Positive":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    end_time = time.time()  # End time
    processing_time = end_time - start_time  # Processing time

    accuracy = lr_model.score(X_test, y_test)  # Accuracy

    greeting = f"The sentiment of your review (Logistic Regression) is {sentiment}, value: {prediction}\n"
    greeting += f"Processing Time: {processing_time:.4f} seconds\n"
    greeting += f"Accuracy: {accuracy:.2f}"

    greeting_label_lr.config(text=greeting)


# Analisis sentimen teks
def classify_sentiment():
    start_time = time.time()  # Memulai pengukuran waktu

    review_text = name.get()
    cleaned_review = preprocess_review(review_text)
    prediction = model.predict(vectorizer.transform([cleaned_review]))[0]
    if prediction == "Positive":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    end_time = time.time()  # Mengakhiri pengukuran waktu
    processing_time = end_time - start_time  # Menghitung waktu pemrosesan

    accuracy = model.score(X_test, y_test)  # Menghitung akurasi prediksi

    greeting = f"The sentiment of your review is {sentiment}, value : {prediction}\n"
    greeting += f"Processing Time: {processing_time:.4f} seconds\n"
    greeting += f"Accuracy: {accuracy:.2f}"

    greeting_label.config(text=greeting)


# Preprocessing
def preprocess_review(review):
    # Case-Folding: Mengubah teks menjadi lowercase
    review = review.lower()

    # Menghapus simbol & angka, menghapus whitespaces, dan mengubah apostrophe/short word
    cleaned_tokens = []
    for token in TextBlob(review).words:
        clean_token = "".join(e for e in token if e.isalpha() or e in ["'", "-"])
        if clean_token:
            clean_token = clean_token.strip()
            corrections = {
                "'s": " is",
                "'re": " are",
                "'ll": " will",
                "'ve": " have",
                "'d": " would",
                "n't": " not",
            }
            clean_token = corrections.get(clean_token, clean_token)
            cleaned_tokens.append(clean_token)

    # Tokenisasi: Memisahkan teks menjadi kata-kata
    tokens = word_tokenize(" ".join(cleaned_tokens))

    # Stopwords Removal: Menghapus kata-kata yang tidak memiliki makna (stopwords)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization: Mengubah kata-kata menjadi bentuk dasarnya
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Mengembalikan teks yang telah dipreproses dalam bentuk string
    return " ".join(tokens)


# Reset
def clear_input():
    name.delete(0, tk.END)
    greeting_label.config(text="")
    greeting_label_nb.config(text="")
    greeting_label_rf.config(text="")
    greeting_label_lr.config(text="")


# Memuat data latih dari file CSV
data = pd.read_csv("hasil_preprocessing.csv")

# Menggunakan kolom "Processed_Review" sebagai teks ulasan
reviews = data["Processed_Review"].values

# Menggunakan kolom "Sentiment_Label" sebagai label sentimen
labels = data["Sentiment_Label"].values

# Menggunakan TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(reviews)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Membuat window
root = tk.Tk()
root.title("Project Styx")

# Membuat label
instruction_label = tk.Label(root, text="Enter Your Review :")
instruction_label.pack()

# Membuat input
name = tk.Entry(root)
name.pack()

# Membuat tombol
button_analisis = tk.Button(root, text="Analyze (KNN)", command=classify_sentiment)
button_analisis.pack()

# Button for Naive Bayes analysis
button_analisis_nb = tk.Button(
    root, text="Analyze (Naive Bayes)", command=classify_sentiment_nb
)
button_analisis_nb.pack()

# Button for Random Forest analysis
button_analisis_rf = tk.Button(
    root, text="Analyze (Random Forest)", command=classify_sentiment_rf
)
button_analisis_rf.pack()

# Button for Logistic Regression analysis
button_analisis_lr = tk.Button(
    root, text="Analyze (Logistic Regression)", command=classify_sentiment_lr
)
button_analisis_lr.pack()

# Membuat tombol reset
button_reset = tk.Button(root, text="Clear", command=clear_input)
button_reset.pack()

# Membuat label untuk hasil
greeting_label = tk.Label(root, text="")
greeting_label.pack()

# Label for Naive Bayes result
greeting_label_nb = tk.Label(root, text="")
greeting_label_nb.pack()

# Label for Random Forest result
greeting_label_rf = tk.Label(root, text="")
greeting_label_rf.pack()

# Label for Logistic Regression result
greeting_label_lr = tk.Label(root, text="")
greeting_label_lr.pack()

# Menjalankan aplikasi
root.mainloop()
