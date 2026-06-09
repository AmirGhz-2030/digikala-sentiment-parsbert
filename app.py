import streamlit as st
import hazm
import re
import pandas as pd
import plotly.express as px
import io
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Deep Learning NLP Dashboard", page_icon="🧠", layout="wide")

# --- استایل‌های سازمانی ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    .stApp { background-color: #0b1120 !important; }
    html, body, [class*="css"], [class*="st-"] { font-family: 'Vazirmatn', sans-serif !important; direction: rtl; text-align: right; color: #e2e8f0 !important; }
    [data-testid="stSidebar"] { background-color: #111827 !important; border-left: 1px solid #1f2937 !important; }
    div[data-testid="metric-container"] { background-color: #1f2937 !important; border: 1px solid #374151 !important; padding: 15px !important; border-radius: 8px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important; }
    div[data-testid="stMetricValue"] { font-size: 22px !important; white-space: normal !important; overflow-wrap: break-word !important; }
    .stTextArea textarea { background-color: #111827 !important; color: #f8fafc !important; border: 1px solid #374151 !important; border-radius: 8px !important; }
    .stButton > button { background-color: #2563eb !important; color: white !important; border-radius: 8px !important; border: none !important; font-weight: bold !important; }
    .stButton > button:hover { background-color: #1d4ed8 !important; }
    .stDownloadButton > button { background-color: #10b981 !important; color: white !important; }
    
    .xai-grid { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; direction: rtl;}
    .xai-card { background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 12px 20px; min-width: 140px; text-align: center; transition: transform 0.2s;}
    .xai-card:hover { transform: translateY(-3px); }
    .xai-word { font-size: 16px; font-weight: bold; color: #f1f5f9; margin-bottom: 5px;}
    .val-positive { color: #10b981; font-family: monospace; font-size: 15px; direction: ltr;}
    .val-negative { color: #ef4444; font-family: monospace; font-size: 15px; direction: ltr;}
    .aspect-badge { background-color: #1e3a8a; color: #60a5fa; padding: 5px 12px; border-radius: 15px; font-size: 13px; font-weight: bold; margin-left: 5px; display: inline-block; border: 1px solid #3b82f6; }
    header { background-color: transparent !important; }
    
    div[data-baseweb="popover"] > div, div[data-testid="stPopoverBody"], div[role="dialog"] { background-color: #1f2937 !important; border: 1px solid #374151 !important; }
    ul[data-testid="main-menu-list"], ul[data-baseweb="menu"], ul[role="listbox"] { background-color: transparent !important; }
    ul[data-testid="main-menu-list"] li, ul[data-baseweb="menu"] li, ul[role="listbox"] li, li[role="option"], div[role="menuitem"] { background-color: #1f2937 !important; border: none !important; }
    div[data-baseweb="popover"] *, div[role="dialog"] *, div[data-baseweb="select"] * { color: #e2e8f0 !important; }
    ul[data-testid="main-menu-list"] li:hover, ul[data-baseweb="menu"] li:hover, ul[role="listbox"] li:hover, li[role="option"]:hover, div[role="menuitem"]:hover { background-color: #374151 !important; cursor: pointer !important; }
    div[data-baseweb="select"] > div { background-color: #111827 !important; border-color: #374151 !important; }
    code { background-color: #111827 !important; color: #93c5fd !important; border: 1px solid #374151 !important; padding: 2px 6px !important; border-radius: 4px !important; }
    
    [data-testid="stFileUploader"] section { background-color: #111827 !important; border: 2px dashed #3b82f6 !important; border-radius: 8px !important; padding: 20px !important; }
    [data-testid="stFileUploader"] section * { color: #cbd5e1 !important; }
    [data-testid="stFileUploader"] section button { background-color: #1f2937 !important; color: #60a5fa !important; border: 1px solid #374151 !important; font-weight: bold !important; }
    [data-testid="stFileUploader"] section button:hover { background-color: #374151 !important; border-color: #93c5fd !important; color: white !important; }
    [data-testid="stDataFrame"] { border: 1px solid #374151 !important; border-radius: 8px !important; overflow: hidden !important; }
    </style>
""", unsafe_allow_html=True)

# --- خواندن پویای اطلاعات مدل ---
trained_model_path = "./dl_sentiment_model"
dynamic_accuracy = "محاسبه نشده"
best_epoch_val = "نامشخص"

metrics_file_path = os.path.join(trained_model_path, "metrics.json")
if os.path.exists(metrics_file_path):
    try:
        with open(metrics_file_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            dynamic_accuracy = f"{meta_data.get('accuracy', 74.5)}%"
            best_epoch_val = f"اپوک {meta_data.get('best_epoch', 2)}"
    except Exception:
        dynamic_accuracy = "خطا در خواندن"

# --- توابع پیش‌پردزاش ---
normalizer = hazm.Normalizer()
def advanced_auto_correct(text):
    if not isinstance(text, str): return ""
    return re.sub(r'(.)\1{2,}', r'\1', text)
def clean_text_dl(text):
    if not isinstance(text, str): return ""
    return normalizer.normalize(text)
def extract_aspects(text):
    aspect_map = {
        "💰 قیمت و ارزش خرید": ["قیمت", "ارزان", "گران", "بها", "هزینه", "پول", "ارزش", "مفت"],
        "🛠️ کیفیت و ساخت": ["کیفیت", "جنس", "ساخت", "مرغوب", "خراب", "بدنه", "پلاستیک", "ضعیف", "محکم"],
        "📦 بسته بندی و ارسال": ["بسته", "پک", "کارتن", "ارسال", "پست", "تحویل", "پیک", "دیر", "زود"],
        "🔋 عملکرد و مشخصات فنی": ["باتری", "شارژ", "دوربین", "صفحه", "سرعت", "داغ", "حافظه", "هنگ"]
    }
    return [asp for asp, keys in aspect_map.items() if any(k in text.lower() for k in keys)]

# --- بارگذاری مدل ---
@st.cache_resource
def load_dl_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(trained_model_path).to(device)
    model.eval()
    return model, tokenizer, device

try:
    model, tokenizer, device = load_dl_model()
except Exception:
    st.error("لطفاً ابتدا فایل جوپیتر را اجرا کنید تا پوشه مدل ساخته شود.")
    st.stop()

# --- الگوریتم XAI ---
def get_word_importances(text, model, tokenizer, device):
    words = text.split()[:40] 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        base_outputs = model(**inputs)
        base_probs = torch.nn.functional.softmax(base_outputs.logits, dim=-1)[0]
        base_pred_class = torch.argmax(base_probs).item()
        base_conf = base_probs[base_pred_class].item()

    importances = {}
    for i, word in enumerate(words):
        masked_text = " ".join(words[:i] + words[i+1:])
        masked_inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            masked_outputs = model(**masked_inputs)
            masked_probs = torch.nn.functional.softmax(masked_outputs.logits, dim=-1)[0]
            masked_conf = masked_probs[base_pred_class].item()
        importance = base_conf - masked_conf
        if base_pred_class == 0: importance = -importance
        importances[word] = importance * 100 
    return importances, base_pred_class, base_probs

# --- سایدبار ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=90)
    st.markdown("### ⚙️ مشخصات معماری سیستم")
    st.info("نوع شبکه:\n\n**Transformer (Bi-Encoder)**")
    st.success(f"سخت‌افزار پردازش: **{str(device).upper()}**")
    st.divider()
    st.caption(" داشبورد هوشمند")
    st.markdown("<h4 style='color: #60a5fa;'>👨‍💻 سازنده: AmirGhz-2030</h4>", unsafe_allow_html=True)

# --- بدنه اصلی ---
st.markdown("<h2 style='color: #60a5fa;'>داشبورد یادگیری عمیق (Deep NLP XAI)</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8;'>طبقه‌بندی ۳ کلاسه (مثبت، منفی، خنثی) با معماری ترانسفورمر و الگوریتم حساسیت.</p>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("معماری شبکه", "ParsBERT", "Transformer") 
col2.metric("استنتاج (Inference)", str(device).upper(), "RTX 3070 🚀" if "cuda" in str(device) else "CPU")
col3.metric("نقطه توقف بهینه", best_epoch_val, "Early Stopping")
col4.metric("دقت مدل (Accuracy)", dynamic_accuracy, "خوانده شده از شناسنامه مدل 📂")
st.divider()

tab1, tab2 = st.tabs(["🎯 تحلیل انفرادی نظر (با ابزار XAI & Aspect)", "📊 پردازش گروهی و کلان (آپلود فایل CSV)"])

with tab1:
    user_input = st.text_area("ورودی متن کاربر:", height=100, placeholder="متن خود را برای پردازش در شبکه عصبی وارد کنید...")
    if st.button("شروع پردازش هوشمند متن"):
        if user_input.strip() == "":
            st.warning("لطفاً متنی وارد کنید.")
        else:
            with st.spinner("در حال واکشی لایه‌های پنهان شبکه عصبی..."):
                corrected_input = advanced_auto_correct(user_input)
                cleaned_input = clean_text_dl(corrected_input)
                importances, pred_class, probs = get_word_importances(cleaned_input, model, tokenizer, device)
                prob_neg, prob_neu, prob_pos = probs[0].item(), probs[1].item(), probs[2].item()
                confidence = max(prob_neg, prob_neu, prob_pos)
                
                if corrected_input != user_input:
                    st.markdown(f"✍️ **اصلاح ساختار املایی:** `{corrected_input}`")
                
                aspects = extract_aspects(user_input)
                st.markdown("#### 🔍 جنبه‌های استخراج شده از متن (Aspects):")
                if aspects:
                    st.markdown("".join([f'<span class="aspect-badge">{asp}</span>' for asp in aspects]), unsafe_allow_html=True)
                else:
                    st.caption("جنبه تخصصی صریحی یافت نشد.")
                
                st.markdown("<br>#### 🎯 نتیجه طبقه‌بندی شبکه عصبی:", unsafe_allow_html=True)
                if pred_class == 2:
                    st.success(f"وضعیت نظر: **مثبت 😊** | ضریب اطمینان: {confidence*100:.1f}%")
                    st.write('<style>[data-testid="stProgress"] > div > div { background-color: #10b981 !important; }</style>', unsafe_allow_html=True)
                elif pred_class == 1:
                    st.info(f"وضعیت نظر: **خنثی 😐** | ضریب اطمینان: {confidence*100:.1f}%")
                    st.write('<style>[data-testid="stProgress"] > div > div { background-color: #64748b !important; }</style>', unsafe_allow_html=True)
                else:
                    st.error(f"وضعیت نظر: **منفی 😡** | ضریب اطمینان: {confidence*100:.1f}%")
                    st.write('<style>[data-testid="stProgress"] > div > div { background-color: #ef4444 !important; }</style>', unsafe_allow_html=True)
                st.progress(float(confidence))
                
                st.markdown("<br>#### 🕵️‍♂️ تحلیل حساسیت شبکه عصبی (Deep XAI):", unsafe_allow_html=True)
                if importances:
                    sorted_words = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
                    html_cards = '<div class="xai-grid">'
                    for word, score in sorted_words:
                        if score > 0.1:
                            html_cards += f'<div class="xai-card"><div class="xai-word">{word}</div><div class="val-positive">+{score:.2f}</div></div>'
                        elif score < -0.1:
                            html_cards += f'<div class="xai-card"><div class="xai-word">{word}</div><div class="val-negative">{score:.2f}</div></div>'
                        else:
                            html_cards += f'<div class="xai-card"><div class="xai-word">{word}</div><div style="color:#94a3b8; font-family:monospace; direction: ltr;">{score:.2f}</div></div>'
                    html_cards += '</div>'
                    st.markdown(html_cards, unsafe_allow_html=True)

with tab2:
    st.markdown("#### 📊 پردازش کلان داده مشتریان (Batch Processing)")
    uploaded_file = st.file_uploader("فایل CSV نظرات را انتخاب کنید", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        possible_cols = ['body', 'comment', 'text', 'متن']
        text_col = next((c for c in batch_df.columns if c.lower() in possible_cols), batch_df.columns[0])
        text_col = st.selectbox("ستون حاوی متن را تایید کنید:", batch_df.columns, index=list(batch_df.columns).index(text_col))
        
        if st.button("🚀 آغاز استنتاج روی CUDA"):
            with st.spinner(f"در حال انتقال تانسورها به حافظه VRAM کارت گرافیک..."):
                texts = batch_df[text_col].astype(str).apply(advanced_auto_correct).apply(clean_text_dl).tolist()
                
                batch_size = 32
                predictions = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                        predictions.extend(preds)
                
                batch_df['predicted_sentiment'] = predictions
                pos_count = sum(p == 2 for p in predictions)
                neu_count = sum(p == 1 for p in predictions)
                neg_count = sum(p == 0 for p in predictions)
                
                label_mapping = {2: 'مثبت', 1: 'خنثی', 0: 'منفی'}
                batch_df['وضعیت نهایی'] = batch_df['predicted_sentiment'].map(label_mapping)
                
                rep_col1, rep_col2 = st.columns([1, 2])
                with rep_col1:
                    st.markdown("##### 📈 گزارش تحلیل توده‌ای:")
                    st.write(f"🔹 حجم داده: **{len(batch_df)} رکورد**")
                    st.write(f"🟢 مثبت: **{pos_count}**")
                    st.write(f"⚪ خنثی: **{neu_count}**")
                    st.write(f"🔴 منفی: **{neg_count}**")
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        batch_df[[text_col, 'وضعیت نهایی']].to_excel(writer, index=False)
                    st.download_button("📥 دانلود نتایج (Excel)", data=buffer.getvalue(), file_name="DeepLearning_3Class_Results.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
                
                with rep_col2:
                    df_chart = pd.DataFrame({"احساس": ["مثبت", "خنثی", "منفی"], "تعداد": [pos_count, neu_count, neg_count]})
                    fig = px.pie(df_chart, names="احساس", values="تعداد", template="plotly_dark", color_discrete_sequence=['#10b981', '#64748b', '#ef4444'])
                    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("##### 🔎 پیش‌نمایش بخشی از نتایج (۱۰ رکورد اول):")
                st.dataframe(batch_df[[text_col, 'وضعیت نهایی']].head(10), use_container_width=True)
