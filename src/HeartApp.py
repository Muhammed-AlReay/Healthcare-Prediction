import streamlit as st
import pandas as pd
import numpy as np
import pickle

# إعداد الصفحة
st.set_page_config(page_title="Medical Prediction System", layout="wide")
st.title("🔬 Medical Prediction System - نظام التنبؤ الطبي")

# تحميل النماذج
heart_model = pickle.load(open("heart_model_model.sav", "rb"))
diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
hypertension_model = pickle.load(open("Hypertensive_model.sav", "rb"))

# إنشاء التبويبات
tab1, tab2, tab3 = st.tabs(["❤ Heart Disease", "🩸 Diabetes", "💢 Hypertension"])

# ---------------- Tab 1: Heart ----------------
with tab1:
    st.subheader("❤ توقع مرض القلب")
    st.markdown("يرجى إدخال بياناتك للتنبؤ بوجود مرض القلب.")

    age = st.slider("العمر", 18, 100, 50)
    sex = st.selectbox("الجنس", ["ذكر", "أنثى"])
    cp = st.selectbox("نوع ألم الصدر (cp)", [0, 1, 2, 3])
    trtbps = st.slider("ضغط الدم أثناء الراحة", 80, 200, 120)
    chol = st.slider("الكوليسترول", 100, 600, 200)
    fbs = st.selectbox("سكر صائم > 120؟", ["لا", "نعم"])
    restecg = st.selectbox("نتائج تخطيط القلب", [0, 1, 2])
    thalachh = st.slider("أعلى معدل ضربات القلب", 70, 210, 150)
    exng = st.selectbox("هل هناك ألم أثناء التمرين؟", ["لا", "نعم"])
    oldpeak = st.slider("الـ ST depression", 0.0, 6.0, 1.0, step=0.1)
    slp = st.selectbox("ميل الـ ST", [0, 1, 2])
    caa = st.selectbox("عدد الأوعية الدموية", [0, 1, 2, 3])
    thall = st.selectbox("حالة thall", [0, 1, 2, 3])

    heart_input = np.array([
        age, 1 if sex == "ذكر" else 0, cp, trtbps, chol,
        1 if fbs == "نعم" else 0, restecg, thalachh,
        1 if exng == "نعم" else 0, oldpeak, slp, caa, thall
    ]).reshape(1, -1)

    if st.button("توقع القلب"):
        result = heart_model.predict(heart_input)[0]
        if result == 1:
            st.error("⚠ هناك احتمال لوجود مرض قلب.")
        else:
            st.success("✅ لا توجد مؤشرات على مرض القلب.")

# ---------------- Tab 2: Diabetes ----------------
with tab2:
    st.subheader("🩸 توقع مرض السكري")
    st.markdown("أدخل بياناتك للتنبؤ بمرض السكري.")

    pregnancies = st.number_input("عدد مرات الحمل", 0, 20, 1)
    glucose = st.slider("مستوى الجلوكوز", 0, 200, 100)
    bp = st.slider("ضغط الدم", 0, 150, 70)
    skin_thickness = st.slider("سمك الجلد", 0, 100, 20)
    insulin = st.slider("مستوى الأنسولين", 0, 900, 80)
    bmi = st.slider("مؤشر كتلة الجسم (BMI)", 0.0, 70.0, 25.0)
    dpf = st.slider("عامل الوراثة (DPF)", 0.0, 2.5, 0.5)
    age_dia = st.slider("العمر", 18, 100, 30)

    diabetes_input = np.array([
        pregnancies, glucose, bp, skin_thickness,
        insulin, bmi, dpf, age_dia
    ]).reshape(1, -1)

    if st.button("توقع السكري"):
        result = diabetes_model.predict(diabetes_input)[0]
        if result == 1:
            st.error("⚠ احتمال الإصابة بمرض السكري.")
        else:
            st.success("✅ لا توجد مؤشرات على مرض السكري.")

# ---------------- Tab 3: Hypertension ----------------
with tab3:
    st.subheader("💢 توقع مرض الضغط")
    st.markdown("أدخل البيانات لتوقع ما إذا كنت تعاني من ضغط دم مرتفع.")

    age_hyp = st.slider("العمر", 18, 100, 40)
    gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
    bmi_hyp = st.slider("BMI", 10.0, 50.0, 22.5)
    smoker = st.selectbox("هل أنت مدخن؟", ["نعم", "لا"])
    activity = st.selectbox("مستوى النشاط البدني", ["منخفض", "متوسط", "مرتفع"])

    hyp_input = np.array([
        age_hyp,
        1 if gender == "ذكر" else 0,
        bmi_hyp,
        1 if smoker == "نعم" else 0,
        {"منخفض": 0, "متوسط": 1, "مرتفع": 2}[activity]
    ]).reshape(1, -1)

    if st.button("توقع الضغط"):
        result = hypertension_model.predict(hyp_input)[0]
        if result == 1:
            st.error("⚠ هناك مؤشرات على ارتفاع ضغط الدم.")
        else:
            st.success("✅ ضغط الدم طبيعي.")

