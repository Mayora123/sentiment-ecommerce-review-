import streamlit as st
import joblib

# Load model
try:
    model = joblib.load("model_svm.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    st.error("❌ File model atau vectorizer tidak ditemukan. Upload dulu!")
    st.stop()

st.set_page_config(page_title="Sentiment Review E-Commerce")

st.title("🛍️ Analisis Sentimen Review E-Commerce")

review = st.text_area("Masukkan review:")

if st.button("Prediksi"):
    if review.strip() == "":
        st.warning("Masukkan review terlebih dahulu.")
    else:
        data = vectorizer.transform([review])
        pred = model.predict(data)[0]
        
        if pred == 1:
            st.success("💚 Sentimen: **POSITIF**")
        else:
            st.error("❤️ Sentimen: **NEGATIF**")
