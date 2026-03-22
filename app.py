from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "burnout_model.pkl"


def inject_tailwind_cdn() -> None:
    """โหลด Tailwind CDN + ฟอนต์ DM Sans (โทนขาวดำ อ่านง่าย)"""
    st.html(
        """
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  theme: {
    extend: {
      fontFamily: {
        sans: ["DM Sans", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
};
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
        """
    )


def inject_streamlit_theme_css() -> None:
    """ธีมขาว–ดำ + ระยะห่างอ่านง่าย + ช่องกรอกสว่าง"""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
html, body, .stApp {
  font-family: "DM Sans", ui-sans-serif, system-ui, sans-serif;
}
.stApp {
  background: linear-gradient(180deg, #fafafa 0%, #f4f4f5 50%, #e4e4e7 100%);
  color: #18181b;
  color-scheme: light;
}
[data-testid="stHeader"] {
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid #e4e4e7;
}
.block-container {
  padding-top: 1.75rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px !important;
}
div[data-testid="column"] {
  background: #ffffff;
  border: 1px solid #e4e4e7;
  border-radius: 1rem;
  padding: 1.35rem 1.25rem 1.6rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
div[data-testid="column"]:hover {
  border-color: #d4d4d8;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.05);
}
div[data-testid="column"] [data-testid="stVerticalBlock"] {
  gap: 1.05rem !important;
}
[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] {
  font-size: 0.9rem !important;
  line-height: 1.45 !important;
  font-weight: 500 !important;
  color: #3f3f46 !important;
  margin-bottom: 0.2rem !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] [data-baseweb="input"] input {
  background-color: #ffffff !important;
  color: #18181b !important;
  border: 1px solid #d4d4d8 !important;
  border-radius: 0.5rem !important;
  min-height: 2.5rem;
}
[data-testid="stNumberInput"] button {
  background: #f4f4f5 !important;
  color: #52525b !important;
  border-color: #d4d4d8 !important;
}
div[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  color: #18181b !important;
  border: 1px solid #d4d4d8 !important;
  border-radius: 0.5rem !important;
  min-height: 2.5rem;
}
.stSlider [data-baseweb="slider"] {
  background-color: #e4e4e7 !important;
  padding-top: 0.35rem;
  padding-bottom: 0.35rem;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
  background-color: #27272a !important;
  border: 2px solid #ffffff !important;
}
/* st.radio — ป้ายตัวเลือก (แนวตั้ง/แนวนอน) ให้เป็นสีดำทั้งหมด */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label span,
[data-testid="stRadio"] p,
[data-testid="stRadio"] span,
[data-testid="stRadio"] [data-baseweb="radio"] ~ div {
  color: #000000 !important;
}
[data-testid="stRadio"] {
  color: #000000 !important;
}
[data-testid="stRadio"] label {
  font-size: 0.9rem !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown p, .stMarkdown label, label {
  color: #18181b !important;
}
.stButton > button[kind="primary"] {
  border-radius: 0.75rem !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  background: #18181b !important;
  color: #fafafa !important;
  border: 1px solid #18181b !important;
  padding: 0.7rem 1.5rem !important;
}
hr {
  border-color: #e4e4e7 !important;
}
/* ตัวเลขสรุปใต้กราฟความน่าจะเป็น (st.metric) — ให้ตัวอักษรอ่านชัดบนพื้นสว่าง */
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] label {
  color: #000000 !important;
}
[data-testid="stMetricValue"] {
  color: #000000 !important;
}
[data-testid="stMetricDelta"] {
  color: #000000 !important;
}
[data-testid="stMetric"] {
  color: #000000 !important;
}
/* st.tabs — ข้อความแท็บทุกแท็บ (เลือก/ไม่เลือก) เป็นสีดำ */
[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] [data-baseweb="tab"] span,
[data-testid="stTabs"] button[role="tab"],
[data-testid="stTabs"] [role="tab"] {
  color: #000000 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: #000000 !important;
  border-bottom-color: #000000 !important;
}
[data-testid="stTabs"] [aria-selected="false"] {
  color: #000000 !important;
}
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] * {
  color: #000000 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def model_has_feature_importances(model: Pipeline) -> bool:
    """Random Forest / Gradient Boosting มี feature_importances_ — SVM / บางโมเดลไม่มี"""
    if not isinstance(model, Pipeline) or "classifier" not in model.named_steps:
        return False
    clf = model.named_steps["classifier"]
    return hasattr(clf, "feature_importances_")


def _importance_detail(model: Pipeline) -> tuple[pd.DataFrame, list[str], list[str]]:
    """คืนค่า DataFrame รายละเอียด + รายชื่อฟีเจอร์ตัวเลข/หมวดหมู่ต้นทาง"""
    if not isinstance(model, Pipeline):
        raise TypeError("คาดหวัง sklearn Pipeline")
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        raise AttributeError(
            "ตัวจำแนกในโมเดลนี้ไม่มี feature_importances_ (เช่น SVM) — ใช้ Random Forest ที่บันทึกจากโน้ตบุ๊ก"
        )

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for name, _, cols in preprocessor.transformers_:
        if name == "remainder" or name.startswith("remainder"):
            continue
        if name == "num":
            numeric_features = [str(c) for c in cols]
        elif name == "cat":
            categorical_features = [str(c) for c in cols]

    cat_enc = preprocessor.named_transformers_["cat"]
    ohe_names = list(cat_enc.get_feature_names_out(categorical_features))
    all_names = list(numeric_features) + ohe_names
    imp = np.asarray(clf.feature_importances_, dtype=float)
    if len(all_names) != len(imp):
        raise ValueError(
            f"จำนวนชื่อฟีเจอร์ ({len(all_names)}) ไม่ตรงกับ feature_importances_ ({len(imp)}) — ไฟล์โมเดลอาจเสียหรือไม่ตรงกับแอป"
        )
    detail = pd.DataFrame({"Feature": all_names, "Importance": imp})
    return detail, numeric_features, categorical_features


def feature_importance_df(model: Pipeline, top_k: int = 18) -> pd.DataFrame:
    """ความสำคัญหลัง one-hot (เรียงจากมากไปน้อย)"""
    detail, _, _ = _importance_detail(model)
    return detail.sort_values("Importance", ascending=False).head(top_k)


def feature_importance_aggregated(model: Pipeline) -> pd.DataFrame:
    """รวมความสำคัญกลับเป็นชื่อตัวแปรต้นฉบับ (รวม one-hot ของแต่ละคอลัมน์หมวดหมู่)"""
    detail, numeric_features, categorical_features = _importance_detail(model)
    detail = detail.copy()
    detail["Feature"] = detail["Feature"].astype(str)
    rows: list[dict] = []
    for col in numeric_features:
        sub = detail[detail["Feature"] == col]
        if len(sub):
            rows.append({"Feature": col, "Importance": float(sub["Importance"].sum())})
    for col in categorical_features:
        prefix = f"{col}_"
        mask = detail["Feature"].str.startswith(prefix, na=False)
        rows.append({"Feature": col, "Importance": float(detail.loc[mask, "Importance"].sum())})
    return pd.DataFrame(rows).sort_values("Importance", ascending=False)


def probability_percent_for_prediction(class_labels: np.ndarray, probabilities: np.ndarray, prediction) -> float:
    """ความน่าจะเป็นของคลาสที่โมเดลเลือก (0–100) — เปรียบเทียบแบบ str เพื่อกัน numpy dtype"""
    pred = str(prediction)
    for lbl, p in zip(class_labels, probabilities):
        if str(lbl) == pred:
            return float(p) * 100.0
    return float(np.max(probabilities)) * 100.0


def probability_figure(class_labels: np.ndarray, probabilities: np.ndarray) -> go.Figure:
    """แท่งแนวนอนแสดงความน่าจะเป็นทุกคลาส (คลิก/โฮเวอร์ได้) — High อยู่บนสุด"""
    display_order = ["High", "Moderate", "Low"]
    color_map = {"Low": "#059669", "Moderate": "#d97706", "High": "#18181b"}
    if len(class_labels) != len(probabilities):
        raise ValueError("จำนวนคลาสกับความน่าจะเป็นไม่ตรงกัน")
    prob_by_class = {str(lbl): float(p) for lbl, p in zip(class_labels, probabilities)}
    probs = [prob_by_class.get(lbl, 0.0) for lbl in display_order]
    colors = [color_map.get(lbl, "#52525b") for lbl in display_order]
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=display_order,
            orientation="h",
            marker_color=colors,
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="outside",
            textfont=dict(color="#000000", size=13),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>ความน่าจะเป็น: %{x:.1%}<extra></extra>",
        )
    )
    axis_font = dict(color="#000000")
    fig.update_layout(
        font=dict(color="#000000"),
        xaxis=dict(
            tickformat=".0%",
            title=dict(text="ความน่าจะเป็น", font=dict(color="#000000")),
            range=[0, 1],
            tickfont=axis_font,
            linecolor="#52525b",
            gridcolor="#e4e4e7",
        ),
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=display_order,
            tickfont=axis_font,
            linecolor="#52525b",
        ),
        height=300,
        margin=dict(l=8, r=48, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafafa",
        showlegend=False,
    )
    return fig


# การตั้งค่าหน้าเว็บ & Disclaimer
st.set_page_config(page_title="Tech Burnout Predictor", page_icon=None, layout="wide")

inject_tailwind_cdn()
inject_streamlit_theme_css()

st.markdown(
    """
<div class="max-w-5xl mx-auto mb-10 rounded-2xl border border-neutral-200 bg-white p-8 sm:p-10 shadow-[0_2px_24px_rgba(0,0,0,0.06)]">
  <div class="flex flex-wrap items-center gap-3 mb-4">
    <span class="inline-flex items-center rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1 text-xs font-semibold uppercase tracking-widest text-neutral-600">
      ML · Mental Health
    </span>
    <span class="hidden sm:inline h-4 w-px bg-neutral-200" aria-hidden="true"></span>
    <span class="text-xs text-neutral-500">Educational demo</span>
  </div>
  <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-neutral-900 mb-3">
    Tech Burnout Predictor
  </h1>
  <p class="text-base sm:text-lg leading-relaxed text-neutral-600 max-w-2xl font-normal">
    แอปพลิเคชันประเมินความเสี่ยงภาวะหมดไฟในการทำงาน (Burnout) สำหรับบุคลากรสาย Tech
  </p>
</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="max-w-5xl mx-auto mb-10 rounded-xl border border-neutral-200 border-l-4 border-l-neutral-900 bg-neutral-50 px-5 py-4 text-sm text-neutral-700 leading-relaxed shadow-sm">
  <p class="font-semibold text-neutral-900 mb-1 tracking-wide uppercase text-xs">Disclaimer</p>
  <p class="font-normal">
    แอปพลิเคชันนี้สร้างขึ้นเพื่อการศึกษาในรายวิชา ML Deployment เท่านั้น ผลการประเมินใช้เพื่อเป็นแนวทางเบื้องต้น
    และไม่สามารถใช้แทนการวินิจฉัยโดยแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพจิตได้
  </p>
</div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "ไม่พบไฟล์ `burnout_model.pkl` — วางไฟล์ไว้ในโฟลเดอร์เดียวกับ `app.py` "
        "หรือรันโน้ตบุ๊ก `Untitled0.ipynb` จนถึงเซลล์บันทึกโมเดลด้วย `joblib.dump`"
    )
    st.stop()
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ (ไฟล์เสียหรือไม่ใช่ Pipeline ที่คาดไว้): {e}")
    st.stop()

if not isinstance(model, Pipeline):
    st.error("ไฟล์โมเดลต้องเป็น sklearn Pipeline ที่มีขั้นตอน `preprocessor` และ `classifier`")
    st.stop()
if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
    st.error("Pipeline ต้องมีชื่อขั้นตอน `preprocessor` และ `classifier` ตามที่เทรนในโน้ตบุ๊ก")
    st.stop()

st.markdown(
    """
<div class="max-w-5xl mx-auto mb-8 flex items-start gap-4">
  <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-neutral-200 bg-neutral-50 text-sm font-bold text-neutral-600 shadow-sm">
  </div>
  <div>
    <h2 class="text-xl sm:text-2xl font-bold tracking-tight text-neutral-900">แบบประเมินความเสี่ยง</h2>
    <p class="mt-1 text-sm text-neutral-500">ปรับค่าให้ตรงกับสถานการณ์จริงของคุณมากที่สุด</p>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
<div class="mb-4 flex items-center gap-2 border-b border-neutral-200 pb-3">
  <span class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-neutral-100 text-xs font-bold text-neutral-600">A</span>
  <p class="text-base font-semibold tracking-wide text-neutral-900">ข้อมูลทั่วไป & งาน</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    age = st.number_input("อายุ (Age)", min_value=18, max_value=80, value=25, help="อายุของคุณ (ปี)")
    gender = st.selectbox("เพศ (Gender)", ["Male", "Female", "Non-binary"])
    job_role = st.selectbox(
        "ตำแหน่งงาน (Job Role)",
        [
            "Backend Developer",
            "Frontend Developer",
            "DevOps",
            "Data Scientist",
            "ML Engineer",
            "Product Manager",
            "QA Engineer",
            "Software Engineer",
        ],
    )
    experience_years = st.number_input("ประสบการณ์ทำงาน (ปี)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
    company_size = st.selectbox("ขนาดบริษัท", ["Startup", "Mid-size", "Large", "MNC"])
    work_mode = st.selectbox("รูปแบบการทำงาน", ["Remote", "Hybrid", "Onsite"])

with col2:
    st.markdown(
        """
<div class="mb-4 flex items-center gap-2 border-b border-neutral-200 pb-3">
  <span class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-neutral-100 text-xs font-bold text-neutral-600">B</span>
  <p class="text-base font-semibold tracking-wide text-neutral-900">ภาระงาน (Workload)</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    work_hours = st.number_input("ชั่วโมงทำงาน/สัปดาห์", min_value=10.0, max_value=120.0, value=40.0, help="ชั่วโมงทำงานปกติต่อสัปดาห์")
    overtime_hours = st.number_input("ชั่วโมง OT/สัปดาห์", min_value=0.0, max_value=80.0, value=5.0)
    meetings = st.slider("จำนวนการประชุม/วัน", min_value=0, max_value=15, value=3)
    deadlines_missed = st.slider("พลาด Deadline ในเดือนที่ผ่านมา", min_value=0, max_value=10, value=0)
    st.markdown('<div class="my-4 h-px w-full bg-neutral-200" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mb-2 text-xs font-semibold uppercase tracking-widest text-neutral-500">การให้คะแนน (1–10)</p>',
        unsafe_allow_html=True,
    )
    job_satisfaction = st.slider("ความพึงพอใจในงาน", 1.0, 10.0, 7.0, help="1=น้อยที่สุด, 10=มากที่สุด")
    manager_support = st.slider("การสนับสนุนจากหัวหน้า", 1.0, 10.0, 6.0)
    work_life_balance = st.slider("Work-Life Balance", 1.0, 10.0, 5.0)

with col3:
    st.markdown(
        """
<div class="mb-4 flex items-center gap-2 border-b border-neutral-200 pb-3">
  <span class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-neutral-100 text-xs font-bold text-neutral-600">C</span>
  <p class="text-base font-semibold tracking-wide text-neutral-900">สุขภาพ & จิตใจ</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    sleep_hours = st.number_input("เวลานอนเฉลี่ย (ชม./วัน)", min_value=2.0, max_value=14.0, value=7.0)
    physical_activity = st.slider("ออกกำลังกาย (วัน/สัปดาห์)", 0, 7, 2)
    screen_time = st.number_input("Screen Time นอกเวลางาน (ชม.)", min_value=0.0, max_value=16.0, value=4.0)
    caffeine_intake = st.slider("ดื่มกาแฟ/เครื่องดื่มคาเฟอีน (แก้ว/วัน)", 0, 10, 2)
    st.markdown('<div class="my-4 h-px w-full bg-neutral-200" aria-hidden="true"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="mb-2 text-xs font-semibold uppercase tracking-widest text-neutral-500">การประเมินตนเอง (1–10)</p>',
        unsafe_allow_html=True,
    )
    social_support = st.slider("การสนับสนุนจากสังคม/เพื่อน", 1.0, 10.0, 6.0)
    stress_level = st.slider("ระดับความเครียดปัจจุบัน", 1.0, 10.0, 5.0)
    anxiety_score = st.slider("ระดับความกังวล", 1.0, 10.0, 4.0)
    depression_score = st.slider("ระดับความซึมเศร้า", 1.0, 10.0, 3.0)
    has_therapy = st.radio("คุณกำลังรับการบำบัดจิตใจหรือไม่?", ["ไม่", "ใช่"])
    has_therapy_val = 1 if has_therapy == "ใช่" else 0

st.divider()

st.caption(
    "หมายเหตุ: หลังกดวิเคราะห์แล้ว คุณสามารถสลับแท็บหรือปรับสไลเดอร์ในกราฟได้โดยผลจะไม่หาย — "
    "ถ้าแก้ค่าในฟอร์มด้านบนแล้วต้องการคำนวณใหม่ ให้กดปุ่มวิเคราะห์อีกครั้ง"
)
st.session_state.setdefault("analysis_done", False)

if st.button(
    "ประมวลผลความเสี่ยง (Analyze Risk)",
    type="primary",
    use_container_width=True,
    key="btn_analyze_risk",
):
    input_data = pd.DataFrame(
        {
            "age": [age],
            "gender": [gender],
            "job_role": [job_role],
            "experience_years": [experience_years],
            "company_size": [company_size],
            "work_mode": [work_mode],
            "work_hours_per_week": [work_hours],
            "overtime_hours": [overtime_hours],
            "meetings_per_day": [meetings],
            "deadlines_missed": [deadlines_missed],
            "job_satisfaction": [job_satisfaction],
            "manager_support": [manager_support],
            "work_life_balance": [work_life_balance],
            "sleep_hours": [sleep_hours],
            "physical_activity_days": [physical_activity],
            "screen_time_hours": [screen_time],
            "caffeine_intake": [caffeine_intake],
            "social_support_score": [social_support],
            "has_therapy": [has_therapy_val],
            "stress_level": [stress_level],
            "anxiety_score": [anxiety_score],
            "depression_score": [depression_score],
        }
    )
    st.session_state["analysis_done"] = True
    st.session_state["last_prediction"] = model.predict(input_data)[0]
    st.session_state["last_proba"] = model.predict_proba(input_data)[0].tolist()

# แสดงผลเมื่อเคยกดวิเคราะห์แล้ว — ไม่พึ่ง st.button (ซึ่งจะเป็น False ทุกครั้งที่รันใหม่หลังคลิกอย่างอื่น)
if st.session_state.get("analysis_done"):
    prediction = st.session_state["last_prediction"]
    probabilities = np.asarray(st.session_state["last_proba"], dtype=float)
    class_labels = model.classes_
    max_prob = probability_percent_for_prediction(class_labels, probabilities, prediction)

    st.markdown(
        """
<div class="mx-auto mb-4 mt-8 max-w-5xl">
  <h2 class="text-2xl font-bold tracking-tight text-neutral-900">ผลการประเมินของคุณ</h2>
</div>
        """,
        unsafe_allow_html=True,
    )
    c_clear, _ = st.columns([1, 3])
    with c_clear:
        if st.button("ล้างผลการประเมิน", key="btn_clear_results"):
            st.session_state["analysis_done"] = False
            if "last_prediction" in st.session_state:
                del st.session_state["last_prediction"]
            if "last_proba" in st.session_state:
                del st.session_state["last_proba"]

    if str(prediction) == "High":
        st.markdown(
            f"""
<div class="relative mx-auto mb-4 max-w-5xl overflow-hidden rounded-2xl border-2 border-neutral-900 bg-neutral-900 p-6 text-neutral-50 shadow-lg">
  <div class="absolute left-0 top-0 h-full w-1 bg-white/30"></div>
  <div class="flex items-start gap-4 pl-2">
    <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-white/20 bg-white/10 text-sm font-bold text-white">H</div>
    <div>
      <p class="text-xs font-semibold uppercase tracking-widest text-neutral-400">ระดับความเสี่ยง</p>
      <p class="mt-1 text-2xl font-bold text-white">สูง (High)</p>
      <p class="mt-3 text-sm leading-relaxed text-neutral-300">
        โมเดลประเมินว่าคุณมีแนวโน้มอยู่ในช่วงความเสี่ยงสูง ควรพิจารณาปรับภาระงาน การพักผ่อน
        และปรึกษาผู้เชี่ยวชาญหากมีอาการสะสม — ผลนี้เป็นเพียงแนวทางเบื้องต้นจากข้อมูลที่กรอก
      </p>
      <p class="mt-4 text-sm text-neutral-400">ความน่าจะเป็นของคลาสที่เลือก: <span class="font-semibold text-white">{max_prob:.1f}%</span></p>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    elif str(prediction) == "Moderate":
        st.markdown(
            f"""
<div class="relative mx-auto mb-4 max-w-5xl overflow-hidden rounded-2xl border-2 border-amber-600/80 bg-amber-50 p-6 text-amber-950 shadow-md">
  <div class="flex items-start gap-4">
    <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-amber-200 bg-white text-sm font-bold text-amber-800">M</div>
    <div>
      <p class="text-xs font-semibold uppercase tracking-widest text-amber-700/80">ระดับความเสี่ยง</p>
      <p class="mt-1 text-2xl font-bold text-amber-950">ปานกลาง (Moderate)</p>
      <p class="mt-3 text-sm leading-relaxed text-amber-900/90">
        มีสัญญาณที่ควรติดตามและดูแลตนเองอย่างสม่ำเสมอ เช่น จัดการเวลางาน นอนหลับพอ และหาคนที่ไว้ใจพูดคุย
      </p>
      <p class="mt-4 text-sm text-amber-800">ความน่าจะเป็นของคลาสที่เลือก: <span class="font-semibold text-amber-950">{max_prob:.1f}%</span></p>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    elif str(prediction) == "Low":
        st.markdown(
            f"""
<div class="relative mx-auto mb-4 max-w-5xl overflow-hidden rounded-2xl border-2 border-emerald-600/50 bg-emerald-50/90 p-6 text-emerald-950 shadow-sm">
  <div class="flex items-start gap-4">
    <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-emerald-200 bg-white text-sm font-bold text-emerald-800">L</div>
    <div>
      <p class="text-xs font-semibold uppercase tracking-widest text-emerald-700/80">ระดับความเสี่ยง</p>
      <p class="mt-1 text-2xl font-bold text-emerald-950">ต่ำ (Low)</p>
      <p class="mt-3 text-sm leading-relaxed text-emerald-900/90">
        โมเดลประเมินในระดับต่ำเมื่อเทียบกับรูปแบบในชุดข้อมูล ยังคงควรดูแลสุขภาพกายใจอย่างต่อเนื่อง
      </p>
      <p class="mt-4 text-sm text-emerald-800">ความน่าจะเป็นของคลาสที่เลือก: <span class="font-semibold text-emerald-950">{max_prob:.1f}%</span></p>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"ได้คลาสที่ไม่คาดไว้จากโมเดล: {prediction!r} — ตรวจสอบไฟล์โมเดลและคอลัมน์ข้อมูล")

    sorted_p = np.sort(probabilities)[::-1]
    margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
    st.markdown(
        f"<p style='font-size:0.9rem;color:#000000;margin-top:0.5rem;'>"
        f"ความมั่นใจเชิงสัมพัทธ์: ระยะห่างระหว่างความน่าจะเป็นอันดับ 1 กับ 2 ≈ "
        f"<strong>{margin * 100:.1f}%</strong> — ถ้าค่าน้อย แปลว่าสองคลาสแรกใกล้เคียงกัน "
        f"ควรตีความด้วยความระมัดระวัง</p>",
        unsafe_allow_html=True,
    )

    tab_prob, tab_imp = st.tabs(["ความน่าจะเป็นทุกคลาส", "ความสำคัญของฟีเจอร์ (โต้ตอบ)"])

    with tab_prob:
        st.markdown(
            "กราฟด้านล่างซูม/เลื่อนได้ และโฮเวอร์ที่แท่งเพื่อดูเปอร์เซ็นต์ — "
            "เรียง **High → Moderate → Low** จากบนลงล่าง"
        )
        fig_prob = probability_figure(class_labels, probabilities)
        st.plotly_chart(fig_prob, use_container_width=True)
        n_cls = len(class_labels)
        mcols = st.columns(max(n_cls, 1))
        for i, lbl in enumerate(class_labels):
            with mcols[i]:
                st.metric(label=str(lbl), value=f"{probabilities[i] * 100:.1f}%")

    with tab_imp:
        if not model_has_feature_importances(model):
            st.info(
                "โมเดลชุดนี้ไม่รองรับกราฟความสำคัญของฟีเจอร์ (ต้องเป็นต้นไม้เช่น Random Forest "
                "ที่บันทึกจากโน้ตบุ๊ก) — ถ้าเปลี่ยนไปใช้ SVM หรือโมเดลอื่น ส่วนนี้จะว่าง"
            )
        else:
            top_k = st.slider(
                "จำนวนฟีเจอร์ที่แสดง (อันดับบน)",
                min_value=5,
                max_value=25,
                value=15,
                step=1,
                key="fi_top_k",
            )
            view_mode = st.radio(
                "มุมมอง",
                ["รวมตามตัวแปรต้นฉบับ (อ่านง่าย)", "ละเอียดหลัง one-hot"],
                horizontal=True,
                help="แบบรวมจะรวมความสำคัญของทุกค่าที่แตกจาก one-hot ของคอลัมน์นั้น",
                key="fi_view_mode",
            )
            try:
                if view_mode.startswith("รวม"):
                    imp_df = feature_importance_aggregated(model).head(top_k)
                else:
                    imp_df = feature_importance_df(model, top_k=top_k)
                if imp_df.empty:
                    st.warning("ไม่มีข้อมูลความสำคัญของฟีเจอร์ให้แสดง")
                else:
                    fig_imp = px.bar(
                        imp_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color_discrete_sequence=["#27272a"],
                    )
                    fig_imp.update_layout(
                        font=dict(color="#000000"),
                        height=max(380, len(imp_df) * 30),
                        yaxis={
                            "categoryorder": "total ascending",
                            "tickfont": dict(color="#000000"),
                            "title": dict(font=dict(color="#000000")),
                            "linecolor": "#52525b",
                        },
                        xaxis={
                            "tickfont": dict(color="#000000"),
                            "title": dict(font=dict(color="#000000")),
                            "linecolor": "#52525b",
                            "gridcolor": "#e4e4e7",
                        },
                        margin=dict(l=8, r=8, t=8, b=8),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#fafafa",
                        showlegend=False,
                    )
                    fig_imp.update_traces(
                        hovertemplate="<b>%{y}</b><br>ความสำคัญ: %{x:.4f}<extra></extra>",
                        textfont=dict(color="#000000"),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    with st.expander("อธิบายสั้นๆ"):
                        st.markdown(
                            "- **Random Forest** ให้ค่าความสำคัญโดยรวมว่าตัวแปรใดถูกใช้แบ่งคลาสบ่อยในการสร้างต้นไม้\n"
                            "- **รวมตามต้นฉบับ** เหมาะอธิบายให้ฟัง (เช่น น้ำหนักรวมของ `job_role` ทุกค่า)\n"
                            "- **หลัง one-hot** เหมาะดูว่าค่าใดของตัวแปรหมวดหมู่มีส่วนมาก"
                        )
            except Exception as e:
                st.warning(f"ไม่สามารถแสดงกราฟความสำคัญของฟีเจอร์ได้: {e}")
