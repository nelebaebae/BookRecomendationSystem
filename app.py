import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load model yang telah dilatih
model = tf.keras.models.load_model("book_recommendation_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()})

# Load dataset buku
books_df = pd.read_csv("books.csv")
ratings_df = pd.read_csv("ratings.csv")

# Mapping user_id dan book_id ke index
user_ids = ratings_df["user_id"].unique()
book_ids = ratings_df["book_id"].unique()
book_id_to_index = {id: i for i, id in enumerate(book_ids)}
index_to_book = {i: id for i, id in enumerate(book_ids)}

# UI Streamlit
st.set_page_config(page_title="📚 Rekomendasi Buku", layout="wide")
st.title("📖 Sistem Rekomendasi Buku")

# Menampilkan daftar buku yang tersedia
st.sidebar.header("Daftar Buku 📚")
st.sidebar.write("Berikut adalah daftar buku yang tersedia:")
st.sidebar.dataframe(books_df[["title", "author"]].head(10))

# Input judul buku
book_title = st.text_input("🔍 Masukkan judul buku:")

# Mencari buku berdasarkan judul yang diinput
if book_title:
    if book_title in books_df["title"].values:
        book_index = books_df[books_df["title"] == book_title].index[0]
        book_vector = np.array([book_index] * len(user_ids))
        user_vector = np.array(list(range(len(user_ids))))

        # Prediksi skor rekomendasi
        scores = model.predict([user_vector, book_vector]).flatten()

        # Mendapatkan 5 buku rekomendasi terbaik
        top_indices = np.argsort(scores)[-5:][::-1]
        recommended_books = [books_df.iloc[i]["title"] for i in top_indices]

        # Menampilkan hasil rekomendasi
        st.subheader("📌 Rekomendasi Buku:")
        for book in recommended_books:
            st.write(f"- {book}")
    else:
        st.warning("⚠️ Buku tidak ditemukan dalam database.")
