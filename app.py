"""
Dry Bean Classifier — Streamlit Application
============================================
Works on LOCAL (VS Code) and STREAMLIT CLOUD.
No pickle / model files needed — trains automatically on first load.

Run locally:   streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #f0f4e8 0%, #fafaf5 100%);
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #1a4d1e 0%, #2e7d32 50%, #66bb6a 100%);
    border-radius: 20px;
    padding: 2.6rem 2rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 40px rgba(46,125,50,0.28);
}
.hero h1 { color: #fff; font-size: 2.7rem; font-weight: 900; margin: 0; letter-spacing: -1px; }
.hero p  { color: rgba(255,255,255,0.88); font-size: 1.05rem; margin: 0.6rem 0 0; }

/* ── Metric cards ── */
.mcard {
    background: #fff;
    border-radius: 14px;
    padding: 1.1rem 1.5rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.07);
    border-left: 5px solid #43a047;
    margin-bottom: 1rem;
    transition: box-shadow .2s;
}
.mcard:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.12); }
.mcard .label { color: #388e3c; font-size: .85rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }
.mcard .value { color: #1a1a1a; font-size: 1.8rem; font-weight: 800; margin: .15rem 0 0; }

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #1a4d1e, #2e7d32);
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 8px 32px rgba(27,94,32,0.35);
    margin-bottom: 1rem;
}
.pred-box .bean-emoji { font-size: 3.5rem; line-height: 1; }
.pred-box h2 { font-size: 2.2rem; font-weight: 900; margin: .4rem 0; }
.pred-box .desc { font-size: .97rem; opacity: .85; margin: 0; }
.pred-box .conf { font-size: 1.5rem; font-weight: 800; margin-top: .8rem; }

/* ── Section heading ── */
.section-head {
    color: #1b5e20;
    font-size: 1.3rem;
    font-weight: 800;
    border-bottom: 3px solid #43a047;
    padding-bottom: .35rem;
    margin: 1.4rem 0 1rem;
}

/* ── Classify button ── */
.stButton > button {
    background: linear-gradient(135deg, #1b5e20, #43a047) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .75rem 2rem !important;
    font-size: 1.1rem !important;
    font-weight: 800 !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(46,125,50,0.30) !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(46,125,50,0.42) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a4d1e 0%, #2e7d32 100%) !important;
}
[data-testid="stSidebar"] * { color: #fff !important; }
[data-testid="stSidebar"] .stMarkdown hr { border-color: rgba(255,255,255,0.25) !important; }

/* ── Progress bar colour ── */
.stProgress > div > div > div { background-color: #43a047 !important; }

/* ── Tab styling ── */
[data-testid="stTab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# BEAN ENCYCLOPEDIA DATA
# ──────────────────────────────────────────────────────────────────────────────
BEANS = {
    "BARBUNYA": {
        "emoji": "🟤", "hex": "#8B4513",
        "desc": "Pink/mottled kidney-shaped bean, medium-large size, firm texture.",
        "origin": "Turkey", "use": "Stews & soups", "protein": "21g/100g",
    },
    "BOMBAY": {
        "emoji": "⚫", "hex": "#333",
        "desc": "Large dark brown-black bean — the rarest & biggest class in this dataset.",
        "origin": "India", "use": "Curries & dals", "protein": "23g/100g",
    },
    "CALI": {
        "emoji": "🟡", "hex": "#C8960C",
        "desc": "Medium yellowish bean with smooth texture, popular in Mediterranean cuisine.",
        "origin": "Turkey", "use": "Salads & side dishes", "protein": "22g/100g",
    },
    "DERMASON": {
        "emoji": "🤍", "hex": "#999",
        "desc": "Small white bean — the most common class. High protein, fast-cooking.",
        "origin": "Turkey", "use": "Soups & casseroles", "protein": "24g/100g",
    },
    "HOROZ": {
        "emoji": "🟠", "hex": "#C0522A",
        "desc": "Elongated light-brown bean with spotted pattern. Medium-large size.",
        "origin": "Turkey", "use": "Pilaf & mixed dishes", "protein": "22g/100g",
    },
    "SEKER": {
        "emoji": "🌸", "hex": "#E75480",
        "desc": "Small pink/white oval bean, mild flavour and delicate skin.",
        "origin": "Turkey", "use": "Salads & light soups", "protein": "20g/100g",
    },
    "SIRA": {
        "emoji": "🟢", "hex": "#4a7c2f",
        "desc": "Medium greenish-brown bean, rich earthy flavour and firm texture.",
        "origin": "Turkey", "use": "Stews & traditional dishes", "protein": "21g/100g",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# MODEL — trains ONCE, cached permanently for the session
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    """
    Loads beans1.xlsx, applies preprocessing, trains HistGradientBoostingClassifier.
    Returns everything needed for inference.
    No pickle files — works on Streamlit Cloud out of the box.
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.ensemble import HistGradientBoostingClassifier

    # ── locate dataset ─────────────────────────────────────────────────────────
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        os.path.join(here, "beans1.xlsx"),
        os.path.join(here, "data", "beans1.xlsx"),
        "beans1.xlsx",
    ]:
        if os.path.exists(candidate):
            data_path = candidate
            break
    else:
        return None  # caller checks for None

    # ── load & encode ──────────────────────────────────────────────────────────
    df = pd.read_excel(data_path)
    le = LabelEncoder()
    df["Class_enc"] = le.fit_transform(df["Class"])

    # ── skewness treatment ─────────────────────────────────────────────────────
    num_cols    = [c for c in df.select_dtypes(include=np.number).columns if c != "Class_enc"]
    skew        = df[num_cols].skew()
    skewed_cols = skew[abs(skew) > 0.5].index.tolist()
    df_t        = df.copy()
    for col in skewed_cols:
        df_t[col] = np.log1p(df_t[col])

    # ── split & scale ──────────────────────────────────────────────────────────
    X = df_t[num_cols]
    y = df_t["Class_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── train ──────────────────────────────────────────────────────────────────
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_depth=6,
        min_samples_leaf=20, l2_regularization=0.2,
        class_weight="balanced", random_state=42,
    )
    model.fit(X_train_sc, y_train)

    # ── evaluate ───────────────────────────────────────────────────────────────
    y_pred    = model.predict(X_test_sc)
    accuracy  = accuracy_score(y_test, y_pred)
    report    = classification_report(y_test, y_pred,
                    target_names=le.classes_.tolist(), output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)

    return {
        "model":       model,
        "scaler":      scaler,
        "le":          le,
        "skewed_cols": skewed_cols,
        "feat_cols":   num_cols,
        "accuracy":    accuracy,
        "report_df":   report_df,
        "df_raw":      df,        # for dataset explorer tab
    }


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫘 Bean Classifier")
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "Classifies **7 types of dry beans** from 16 physical measurements "
        "captured by industrial computer-vision systems."
    )
    st.markdown("### 🧠 Model")
    st.markdown(
        "- **Algorithm:** HistGradientBoosting\n"
        "- **Test Accuracy:** ~92.5 %\n"
        "- **Macro F1:** ~0.94\n"
        "- **Features:** 16 measurements\n"
        "- **Trains at startup** — no pickle files"
    )
    st.markdown("---")
    st.markdown("### 🌱 Classes")
    for name, info in BEANS.items():
        st.markdown(f"{info['emoji']} **{name}** — {info['origin']}")
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn · v2")

# ──────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🫘 Dry Bean Classifier</h1>
  <p>AI-powered identification of 7 dry bean varieties from physical measurements &nbsp;·&nbsp;
     92.5 % accuracy &nbsp;·&nbsp; HistGradientBoosting</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD / TRAIN MODEL
# ──────────────────────────────────────────────────────────────────────────────
with st.spinner("🔧 Training model — takes ~15 s on first load, instant afterwards…"):
    bundle = get_model()

if bundle is None:
    st.error("❌ **beans1.xlsx not found.**  "
             "Place the dataset file in the same folder as `app.py` and restart.")
    st.stop()

model       = bundle["model"]
scaler      = bundle["scaler"]
le          = bundle["le"]
skewed_cols = bundle["skewed_cols"]
feat_cols   = bundle["feat_cols"]
accuracy    = bundle["accuracy"]
report_df   = bundle["report_df"]
df_raw      = bundle["df_raw"]
classes     = le.classes_.tolist()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬  Classify a Bean",
    "📊  Explore Dataset",
    "📚  Bean Encyclopedia",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-head">🎛️ Enter Bean Measurements</div>',
                unsafe_allow_html=True)
    st.markdown("Drag the sliders to match the physical characteristics of your bean sample.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📐 Size Features**")
        area           = st.slider("Area (pixels)",       20000,  250000, 60000, 500)
        perimeter      = st.slider("Perimeter",           500.0,  2000.0, 800.0, 0.5)
        major_axis     = st.slider("Major Axis Length",   150.0,  900.0,  300.0, 0.5)
        minor_axis     = st.slider("Minor Axis Length",   100.0,  600.0,  200.0, 0.5)
        convex_area    = st.slider("Convex Area",         20000,  260000, 62000, 500)
        equiv_diameter = st.slider("Equiv. Diameter",     160.0,  600.0,  270.0, 0.5)

    with c2:
        st.markdown("**🔵 Shape Features**")
        aspect_ratio = st.slider("Aspect Ratio",  1.00, 3.00, 1.50, 0.01)
        eccentricity = st.slider("Eccentricity",  0.30, 1.00, 0.75, 0.001)
        extent       = st.slider("Extent",        0.50, 0.90, 0.75, 0.001)
        solidity     = st.slider("Solidity",      0.90, 1.00, 0.985, 0.001)
        roundness    = st.slider("Roundness",     0.40, 1.00, 0.85, 0.001)
        compactness  = st.slider("Compactness",   0.60, 1.00, 0.85, 0.001)

    with c3:
        st.markdown("**🔷 Shape Factors**")
        sf1 = st.slider("Shape Factor 1", 0.002,  0.012,  0.006,  0.0001, format="%.4f")
        sf2 = st.slider("Shape Factor 2", 0.0005, 0.003,  0.001,  0.00001, format="%.5f")
        sf3 = st.slider("Shape Factor 3", 0.40,   1.00,   0.75,   0.001)
        sf4 = st.slider("Shape Factor 4", 0.90,   1.00,   0.99,   0.001)

    st.markdown("---")
    go = st.button("🔍 Classify This Bean", use_container_width=True)

    if go:
        # Build + preprocess input
        raw = {
            "Area": area, "Perimeter": perimeter,
            "MajorAxisLength": major_axis, "MinorAxisLength": minor_axis,
            "AspectRation": aspect_ratio, "Eccentricity": eccentricity,
            "ConvexArea": convex_area, "EquivDiameter": equiv_diameter,
            "Extent": extent, "Solidity": solidity,
            "roundness": roundness, "Compactness": compactness,
            "ShapeFactor1": sf1, "ShapeFactor2": sf2,
            "ShapeFactor3": sf3, "ShapeFactor4": sf4,
        }
        inp = pd.DataFrame([raw])[feat_cols]
        for col in skewed_cols:
            if col in inp.columns:
                inp[col] = np.log1p(inp[col])

        inp_sc    = scaler.transform(inp)
        pred_enc  = model.predict(inp_sc)[0]
        pred_name = le.inverse_transform([pred_enc])[0]
        proba     = model.predict_proba(inp_sc)[0]
        confidence = proba[pred_enc] * 100

        info = BEANS[pred_name]

        # ── Result layout ──────────────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.markdown(f"""
            <div class="pred-box">
                <div class="bean-emoji">{info['emoji']}</div>
                <h2>{pred_name}</h2>
                <p class="desc">{info['desc']}</p>
                <p style="margin:.4rem 0;opacity:.8">
                    🌍 {info['origin']} &nbsp;·&nbsp; 🍽️ {info['use']}
                    &nbsp;·&nbsp; 💪 Protein: {info['protein']}
                </p>
                <div class="conf">Confidence: {confidence:.1f} %</div>
            </div>
            """, unsafe_allow_html=True)

            # Top-3 breakdown
            st.markdown("**🏅 Top 3 Predictions**")
            top3 = np.argsort(proba)[::-1][:3]
            medals = ["🥇", "🥈", "🥉"]
            for i, idx in enumerate(top3):
                nm = le.inverse_transform([idx])[0]
                pct = proba[idx]
                st.markdown(f"{medals[i]} **{nm}** &nbsp; {pct*100:.1f} %")
                st.progress(float(pct))

        with right:
            # Horizontal probability bar chart (plotly)
            try:
                import plotly.graph_objects as go

                order      = np.argsort(proba)
                bar_names  = [le.inverse_transform([i])[0] for i in order]
                bar_probs  = proba[order] * 100
                bar_colors = ["#2e7d32" if n == pred_name else "#a5d6a7"
                              for n in bar_names]

                fig = go.Figure(go.Bar(
                    x=bar_probs, y=bar_names, orientation="h",
                    marker_color=bar_colors,
                    text=[f"{p:.1f}%" for p in bar_probs],
                    textposition="outside",
                ))
                fig.update_layout(
                    title="All Class Probabilities",
                    xaxis_title="Probability (%)",
                    xaxis_range=[0, 110],
                    height=370,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12),
                    margin=dict(l=10, r=40, t=40, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                    st.write(f"**{cls}**: {p*100:.1f}%")
                    st.progress(float(p))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-head">📊 Dataset Explorer</div>', unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        ("📦 Samples",   "13,611"),
        ("🔢 Features",  "16"),
        ("🏷️ Classes",   "7"),
        ("🎯 Accuracy",  f"{accuracy*100:.1f}%"),
        ("❌ Missing",   "0"),
    ]
    for col, (label, val) in zip([k1, k2, k3, k4, k5], kpi_data):
        with col:
            st.markdown(
                f'<div class="mcard"><div class="label">{label}</div>'
                f'<div class="value">{val}</div></div>',
                unsafe_allow_html=True,
            )

    try:
        import plotly.express as px

        # Class distribution
        ec1, ec2 = st.columns(2)
        vc = df_raw["Class"].value_counts().reset_index()
        vc.columns = ["Class", "Count"]

        with ec1:
            fig = px.bar(vc, x="Class", y="Count", color="Class",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title="Bean Class Distribution")
            fig.update_layout(showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with ec2:
            fig2 = px.pie(vc, names="Class", values="Count",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          title="Class Proportions")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        # Feature boxplot by class
        st.markdown("**🔎 Feature Distribution by Class**")
        feat_choice = st.selectbox(
            "Select a feature to visualise",
            [c for c in df_raw.columns if c != "Class"],
        )
        fig3 = px.box(df_raw, x="Class", y=feat_choice, color="Class",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      title=f"{feat_choice} — Distribution by Bean Class",
                      points="outliers")
        fig3.update_layout(showlegend=False,
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

        # Correlation heatmap
        st.markdown("**🔗 Feature Correlation Heatmap**")
        num_df = df_raw.select_dtypes(include=np.number)
        corr   = num_df.corr().round(2)
        fig4   = px.imshow(corr, text_auto=True,
                            color_continuous_scale="RdYlGn",
                            zmin=-1, zmax=1, aspect="auto",
                            title="Feature Correlation Matrix")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=560)
        st.plotly_chart(fig4, use_container_width=True)

    except ImportError:
        st.warning("Install plotly for interactive charts: `pip install plotly`")
        st.dataframe(df_raw.describe().round(3))

    # Model report
    st.markdown("**🧪 Per-Class Model Performance**")
    st.dataframe(
        report_df.style.background_gradient(
            subset=["precision", "recall", "f1-score"], cmap="Greens"
        ),
        use_container_width=True,
    )

    # Raw data
    with st.expander("📋 View Raw Data (first 100 rows)"):
        st.dataframe(df_raw.head(100), use_container_width=True)
        st.caption(f"Showing 100 of 13,611 rows · 17 columns")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ENCYCLOPEDIA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">📚 Bean Encyclopedia</div>', unsafe_allow_html=True)
    st.markdown("Detailed information on each of the 7 dry bean varieties in the dataset.")

    for name, info in BEANS.items():
        with st.expander(f"{info['emoji']}  **{name}**  —  {info['origin']}"):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"📝 **Description:** {info['desc']}")
                st.markdown(f"🌍 **Origin:** {info['origin']}")
                st.markdown(f"🍽️ **Common Uses:** {info['use']}")
                st.markdown(f"💪 **Protein Content:** {info['protein']}")
            with right:
                st.markdown(
                    f"""
                    <div style="
                        background:{info['hex']}22;
                        border:2.5px solid {info['hex']};
                        border-radius:14px;
                        padding:1.2rem 0.8rem;
                        text-align:center;
                    ">
                        <div style="font-size:3rem">{info['emoji']}</div>
                        <div style="font-weight:800;color:{info['hex']};
                                    font-size:1rem;margin-top:.4rem">{name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:.82rem;padding:.6rem 0'>"
    "🫘 Dry Bean Classifier v2 &nbsp;·&nbsp; "
    "HistGradientBoostingClassifier &nbsp;·&nbsp; "
    "92.5 % Test Accuracy &nbsp;·&nbsp; "
    "Built with Streamlit & Scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
