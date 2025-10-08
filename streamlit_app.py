import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import numpy as np
import matplotlib.pyplot as plt
import torch

# =============================================
# KONFIGURASI APLIKASI
# =============================================
st.set_page_config(page_title="Analisis Sentimen Indonesia", page_icon="🇮🇩", layout="centered")
st.title("🇮🇩 Analisis Sentimen Bahasa Indonesia – Versi Stabil CPU")

st.markdown("""
🔍 **Aplikasi Analisis Sentimen Otomatis**
- Menggunakan model *Bahasa Indonesia* dan *multilingual*
- Dapat berjalan di CPU tanpa GPU
- Sudah diperbaiki agar tidak selalu netral
""")

# =============================================
# LOAD MODEL (STABIL UNTUK CPU)
# =============================================
@st.cache_resource
def load_models():
    models = []
    model_list = [
        "w11wo/indonesian-roberta-base-sentiment-classifier",  # 🇮🇩 Roberta
        "cardiffnlp/twitter-xlm-roberta-base-sentiment"        # 🌍 Multilingual
    ]

    for m in model_list:
        try:
            st.info(f"🔄 Memuat model: {m}")
            tokenizer = AutoTokenizer.from_pretrained(m)
            model = AutoModelForSequenceClassification.from_pretrained(m, torch_dtype=torch.float32)
            pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=-1)
            models.append(pipe)
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat model: {m}\n{e}")

    if not models:
        st.error("❌ Tidak ada model berhasil dimuat. Pastikan koneksi internet aktif.")
    else:
        st.success(f"✅ {len(models)} model berhasil dimuat.")
    return models


# =============================================
# BOOSTER LOGIKA (agar hasil tidak netral terus)
# =============================================
def sentiment_booster(text, label):
    t = text.lower()
    positif = ["bagus", "hebat", "senang", "puas", "keren", "mantap", "suka", "terbaik", "menyenangkan"]
    negatif = ["buruk", "jelek", "kecewa", "tidak puas", "marah", "parah", "mengecewakan", "benci", "sial"]

    if any(k in t for k in positif):
        return "positive"
    if any(k in t for k in negatif):
        return "negative"
    if "tidak" in t and any(x in t for x in ["bagus", "puas", "senang", "suka"]):
        return "negative"
    if "tidak" in t and any(x in t for x in ["buruk", "jelek", "kecewa"]):
        return "positive"
    return label


# =============================================
# ENSEMBLE + RULE BASED
# =============================================
def ensemble_predict(text, models):
    preds, scores = [], []
    for model in models:
        try:
            res = model(text)[0]
            preds.append(res['label'].lower())
            scores.append(res['score'])
        except Exception:
            preds.append("neutral")
            scores.append(0.33)

    final_label = max(set(preds), key=preds.count)
    avg_score = round(np.mean(scores) * 100, 2)
    final_label = sentiment_booster(text, final_label)
    return final_label, avg_score, preds, scores


# =============================================
# INPUT DAN OUTPUT STREAMLIT
# =============================================
text = st.text_area("Masukkan kalimat Bahasa Indonesia:", "Saya kecewa dengan pelayanan di toko ini.")

if st.button("🔍 Analisis Sentimen"):
    if not text.strip():
        st.warning("Masukkan kalimat terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            models = load_models()
            if models:
                label, score, preds, scores = ensemble_predict(text, models)
                if label == "positive":
                    st.success(f"✅ Sentimen: **Positif** ({score}%)")
                elif label == "negative":
                    st.error(f"❌ Sentimen: **Negatif** ({score}%)")
                else:
                    st.info(f"⚖️ Sentimen: **Netral** ({score}%)")

                st.markdown("### 🔬 Detail Per Model")
                for i, (p, s) in enumerate(zip(preds, scores), 1):
                    st.write(f"Model {i}: {p.capitalize()} — {round(s*100,2)}%")

                fig, ax = plt.subplots(figsize=(5,2.8))
                ax.bar([f"Model {i}" for i in range(1, len(scores)+1)], [s*100 for s in scores],
                       color=["#4CAF50", "#2196F3"])
                ax.set_ylim(0,100)
                ax.set_ylabel("Confidence (%)")
                ax.set_title("Perbandingan Confidence Tiap Model")
                for i, s in enumerate(scores):
                    ax.text(i, s*100 + 2, f"{round(s*100,1)}%", ha="center")
                st.pyplot(fig)

st.markdown("---")
st.caption("🧠 Model: IndoRoberta 🇮🇩 + XLM-Roberta 🌍 + Booster Kata Kunci • Optimized for CPU")
