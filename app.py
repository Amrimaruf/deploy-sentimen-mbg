import streamlit as st
import torch
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 🔹 FUNGSI TAMBAHAN: BACKGROUND IMAGE
# ==========================================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.55);
        z-index: 0;
    }}

    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.4);
    }}

    .stTextArea textarea {{
        background: rgba(255,255,255,0.95) !important;
        color: black !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        border: 2px solid rgba(255,255,255,0.5) !important;
    }}

    .stButton button {{
        background-color: rgba(255, 255, 255, 0.90) !important;
        color: black !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        border: none !important;
    }}

    .stButton button:hover {{
        background-color: black !important;
        color: white !important;
        transform: scale(1.02);
        transition: 0.2s;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==========================================
# 🔹 PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Prediksi Sentimen MBG (IndoBERT)",
    page_icon="🧠",
    layout="centered"
)

add_bg_from_local("assets/mbg.jpg")

# ==========================================
# 🔹 LOAD INDOBERT MODEL
# ==========================================
@st.cache_resource
def load_model():
    MODEL_DIR = "indobert_sentiment_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

# Label mapping (sesuaikan jika perlu)
LABEL_MAP = {0: "Negatif", 1: "Positif"}

# ==========================================
# 🔹 STREAMLIT UI
# ==========================================
st.title("🧠 Prediksi Sentimen Program Makan Bergizi Gratis")
st.write(
    "Masukkan kalimat di bawah untuk melihat prediksi sentimen "
    "menggunakan **model IndoBERT**."
)

user_input = st.text_area("📝 Teks", height=150)

# ==========================================
# 🔹 PREDIKSI
# ==========================================
if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()

        label = LABEL_MAP[pred_id]

        st.subheader("📊 Hasil Prediksi")
        if label == "Positif":
            st.success(f"🟢 **Positif** (Confidence: {confidence:.4f})")
        else:
            st.error(f"🔴 **Negatif** (Confidence: {confidence:.4f})")

# ==========================================
# 🔹 FOOTER
# ==========================================
st.markdown("---")
st.caption("Model: IndoBERT | Fine-tuned untuk Analisis Sentimen Program MBG")
