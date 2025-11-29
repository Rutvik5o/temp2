# app.py
# Combined: PowerBI-style Dark Churn Dashboard + RAG Q&A with Gemini (optional)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import textwrap

# Optional Gemini client - import lazily (so app runs if package missing)
try:
    import google.generativeai as genai  # pip: google-generativeai
    _HAS_GEMINI = True
except Exception:
    genai = None
    _HAS_GEMINI = False

# TF-IDF retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------- PAGE CONFIG -------------------------------------
st.set_page_config(
    page_title="Customer Churn Analyzer — PowerBI Dark + RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- COLOR PALETTE ----------------------------------
PALETTE = px.colors.qualitative.Plotly
ACCENT = "#0ea5a3"

# ------------------------- DARK THEME CSS ---------------------------------
CSS = r'''
<style>
:root{--bg:#0b1220;--card:#0f1724;--muted:#9aa4b2;--accent:#0ea5a3;--accent-2:#06b6d4;}
[data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg,#071019 0%, #071724 40%, #0a1522 100%);
  color: #e6eef6;
  padding-top: 0.6rem;
  padding-bottom: 1rem;
}
.header-card {
  background: linear-gradient(90deg,#071126 0%, #0d2330 40%, var(--accent) 100%);
  color: white;
  padding: 18px;
  border-radius: 10px;
  box-shadow: 0 12px 40px rgba(2,6,23,0.6);
  border: 1px solid rgba(255,255,255,0.03);
}
.header-sub {color: rgba(255,255,255,0.9); opacity:0.95}
section[data-testid="stSidebar"] .css-1lcbmhc{ background: linear-gradient(180deg,#07101a,#0b1420); border-radius:10px; padding:12px; color: #dfe9f2 }
.metric-card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:14px; box-shadow: 0 6px 20px rgba(2,8,23,0.6); border-left:6px solid var(--accent);}
.metric-sub {color:var(--muted); font-size:13px}
.metric-value {font-size:1.45rem; font-weight:700; color: #e6f7f2}
.info-card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:10px; padding:12px;}
.small-muted {color:var(--muted); font-size:13px}
.stButton>button{ background: linear-gradient(90deg,var(--accent),var(--accent-2)); color:white; border:none; padding:8px 18px; border-radius:10px; font-weight:600; box-shadow: 0 6px 20px rgba(14,165,163,0.18); }
.chart-card { padding:10px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005)); border-radius:10px; box-shadow: 0 10px 30px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.02); }
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

# ------------------------- HEADER -----------------------------------------
st.markdown(
    '''
    <div class="header-card">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
        <div>
          <h1 style="margin:0;font-size:1.6rem;">📊 Customer Churn Analyzer — Dark PowerBI + RAG</h1>
          <div class="header-sub" style="font-size:0.95rem;margin-top:6px;">Interactive • Visuals • Gemini RAG (optional)</div>
        </div>
        <div style="text-align:right; font-size:0.85rem; color:rgba(255,255,255,0.85);">
          <div>Built with Streamlit + Plotly</div>
        </div>
      </div>
    </div>
    ''',
    unsafe_allow_html=True
)

# ------------------------- UPLOAD AREA ------------------------------------
col_left, col_right = st.columns([2, 1])
with col_left:
    uploaded = st.file_uploader("📁 Upload CSV (Telco-style)", type=["csv"], key="uploader_combined")
    st.markdown('<div class="small-muted">CSV should include customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,PaymentMethod,Churn (map in sidebar)</div>', unsafe_allow_html=True)
with col_right:
    st.markdown('<div class="info-card"><strong>⚡ Quick Tips</strong><br>• Ensure Tenure & MonthlyCharges are numeric<br>• Churn: Yes/No or 1/0<br>• Optional: set GEMINI_API_KEY in Streamlit secrets to enable Gemini answers</div>', unsafe_allow_html=True)

# ------------------------- LOAD DATA -------------------------------------
SAMPLE = "/mnt/data/chrundata.csv"
if uploaded is None:
    if os.path.exists(SAMPLE):
        try:
            df = pd.read_csv(SAMPLE)
            st.success("✅ Loaded sample dataset")
        except Exception:
            st.error("❌ Sample dataset error. Please upload a CSV file.")
            st.stop()
    else:
        st.info("No sample dataset found. Please upload your CSV to get started.")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Loaded {:,} rows".format(len(df)))
    except Exception as e:
        st.error("❌ Upload error: {}".format(e))
        st.stop()

if df is None or df.empty:
    st.stop()

# ------------------------- SIDEBAR CONFIG --------------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configure Analysis (Dark)")
cols = df.columns.tolist()

def safe_index(lst, value, offset=0):
    try:
        return lst.index(value) + offset
    except ValueError:
        return 0

cust_id_col = st.sidebar.selectbox("👤 Customer ID", ["(none)"] + cols, index=safe_index(cols, "customerID", 1))
churn_col = st.sidebar.selectbox("❌ Churn Flag", ["(none)"] + cols, index=safe_index(cols, "Churn", 1))
tenure_col = st.sidebar.selectbox("📅 Tenure (months)", ["(none)"] + cols, index=safe_index(cols, "tenure", 1))
signup_col = st.sidebar.selectbox("📆 Signup Date (optional)", ["(none)"] + cols, index=safe_index(cols, "signup_date", 1))
last_col = st.sidebar.selectbox("📆 Last Active (optional)", ["(none)"] + cols, index=safe_index(cols, "last_active", 1))

candidate_segments = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "gender", "SeniorCitizen"]
possible_segments = [c for c in candidate_segments if c in cols]
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filters")
segment_filters = {}
for i, s in enumerate(possible_segments):
    vals = sorted(df[s].dropna().unique().tolist())
    if vals:
        default_vals = vals[:min(3, len(vals))]
        segment_filters[s] = st.sidebar.multiselect("🔧 {}".format(s), options=vals, default=default_vals, key="seg_power_{}".format(i))

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Analysis")

# ------------------------- HELPERS ---------------------------------------
def normalize_churn(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    s = series.astype(str).str.strip().str.lower()
    return s.map({'yes':1, 'y':1, 'true':1, '1':1, 't':1, 'churn':1, 'no':0, 'n':0, 'false':0, '0':0, 'stay':0, 'retained':0}).astype(float)

def dark_plotly_layout(fig, height=420, showlegend=True, rotate_x=False):
    fig.update_layout(
        template=None,
        paper_bgcolor='rgba(11,18,32,0)',
        plot_bgcolor='rgba(11,18,32,0)',
        font=dict(color='#e6eef6'),
        height=height,
        legend=dict(bgcolor='rgba(255,255,255,0.02)') if showlegend else dict(visible=False),
        margin=dict(l=60, r=30, t=50, b=120)
    )
    try:
        if rotate_x:
            fig.update_xaxes(tickangle=-45, automargin=True, showgrid=False, zeroline=False, showline=True, linecolor='rgba(255,255,255,0.06)', tickfont=dict(color='#d0e8e2'))
        else:
            fig.update_xaxes(automargin=True, showgrid=False, zeroline=False, showline=True, linecolor='rgba(255,255,255,0.06)', tickfont=dict(color='#d0e8e2'))
        fig.update_yaxes(automargin=True, showgrid=True, gridcolor='rgba(255,255,255,0.03)', zeroline=False, showline=True, linecolor='rgba(255,255,255,0.06)', tickfont=dict(color='#d0e8e2'))
    except Exception:
        pass
    return fig

# ------------------------- RAG Helpers -----------------------------------
# Performance-safe RAG defaults
MAX_RAG_ROWS = 2000
MAX_PASSAGE_CHARS = 1200
TFIDF_MAX_FEATURES = 5000

def row_to_text(row: pd.Series):
    parts = []
    for k, v in row.items():
        if pd.isna(v):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            parts.append(f"{k}: {v}")
        else:
            s = str(v).strip()
            if s:
                parts.append(f"{k}: {s}")
    return " | ".join(parts)

def build_corpus_from_df_sampled(df_local, max_rows=MAX_RAG_ROWS, max_chars=MAX_PASSAGE_CHARS):
    if df_local is None or df_local.shape[0] == 0:
        return []
    n = len(df_local)
    if n <= max_rows:
        sample_df = df_local
    else:
        head_n = min(300, max_rows // 10)
        tail_n = max_rows - head_n
        head_df = df_local.head(head_n)
        tail_df = df_local.sample(n=tail_n, random_state=42)
        sample_df = pd.concat([head_df, tail_df], ignore_index=True)
    passages = []
    for i in range(len(sample_df)):
        txt = row_to_text(sample_df.iloc[i])
        if not txt:
            continue
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " ...[truncated]"
        passages.append(txt)
    return passages

def build_tfidf_index_safe(passages, max_features=TFIDF_MAX_FEATURES):
    if not passages:
        return None, None
    vect = TfidfVectorizer(stop_words="english", max_features=max_features)
    try:
        mat = vect.fit_transform(passages)
    except Exception as e:
        st.error(f"TF-IDF build failed: {e}")
        return None, None
    return vect, mat

def retrieve_top_k_from_index(query, vect, mat, passages, k=3):
    if vect is None or mat is None:
        return []
    qv = vect.transform([query])
    sims = cosine_similarity(qv, mat).flatten()
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for idx in top_idx:
        if sims[idx] <= 0:
            continue
        results.append((float(sims[idx]), passages[idx], idx))
    return results

def extractive_summary_from_passages(passages, question, max_sentences=3):
    q_tokens = set([t.lower() for t in question.split() if len(t) > 2])
    scored_sentences = []
    for p in passages:
        sents = [s.strip() for s in p.replace("|", ". ").split(".") if s.strip()]
        for s in sents:
            stokens = set([t.lower() for t in s.split() if len(t) > 2])
            score = len(q_tokens.intersection(stokens))
            if any(ch.isdigit() for ch in s):
                score += 0.2
            if score > 0:
                scored_sentences.append((score, s))
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored_sentences[:max_sentences]]
    return " ".join(top) if top else None

def init_gemini():
    if not _HAS_GEMINI:
        return False
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return True
    except Exception:
        return False

def call_gemini_generate(context_text: str, user_question: str):
    if not _HAS_GEMINI:
        return None
    ok = init_gemini()
    if not ok:
        return None
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
You are a customer churn analytics assistant.
Use ONLY the context below — do NOT hallucinate. Answer concisely and reference context.

CONTEXT:
{context_text}

QUESTION:
{user_question}

Provide:
1) Direct answer
2) Short reasoning referencing context
3) 2–3 actionable insights.
"""
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else str(response)
    except Exception:
        return None

# ------------------------- MAIN: RUN ANALYSIS ----------------------------------
if run_analysis:
    # compute filtered df2 and store in session_state (so RAG uses it)
    df2 = df.copy()
    for s, selvals in segment_filters.items():
        if selvals:
            try:
                df2 = df2[df2[s].isin(selvals)]
            except Exception:
                st.warning(f"Filter {s} could not be applied (type mismatch).")

    # persist filtered df
    st.session_state["filtered_df"] = df2.copy()
    # clear cached RAG index so it rebuilds for new filter
    for k in ["rag_passages", "rag_vect", "rag_mat", "_rag_source_sig"]:
        st.session_state.pop(k, None)

    st.markdown(f'<div class="info-card"><strong>📋 Filtered Dataset</strong> • {len(df2)} rows</div>', unsafe_allow_html=True)
    st.dataframe(df2.head(8), use_container_width=True, height=200)

    # normalize churn
    churn_key = None
    churn_rate = None
    if churn_col != "(none)":
        try:
            df2["_churn_mapped"] = normalize_churn(df2[churn_col])
            churn_key = "_churn_mapped"
            if df2[churn_key].dropna().shape[0] > 0:
                churn_rate = float(df2[churn_key].dropna().mean())
        except Exception:
            churn_key = None

    # numeric coercion
    total_customers = len(df2)
    avg_tenure = None
    if tenure_col != "(none)":
        df2[tenure_col] = pd.to_numeric(df2[tenure_col], errors="coerce")
        avg_tenure = df2[tenure_col].dropna().mean() if df2[tenure_col].dropna().shape[0] > 0 else None

    if "MonthlyCharges" in df2.columns:
        df2["MonthlyCharges"] = pd.to_numeric(df2["MonthlyCharges"], errors="coerce")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        k1_html = f'<div class="metric-card"><div class="metric-sub">👥 Total Customers</div><div class="metric-value">{total_customers:,}</div></div>'
        st.markdown(k1_html, unsafe_allow_html=True)
    with k2:
        if churn_rate is not None:
            st.markdown(f'<div class="metric-card"><div class="metric-sub">🔴 Churn Rate</div><div class="metric-value">{churn_rate:.1%}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-card"><div class="metric-sub">🔴 Churn Rate</div><div class="metric-value">N/A</div></div>', unsafe_allow_html=True)
    with k3:
        avg_tenure_fmt = "N/A" if avg_tenure is None or (isinstance(avg_tenure, float) and np.isnan(avg_tenure)) else f"{avg_tenure:.1f}"
        st.markdown(f'<div class="metric-card"><div class="metric-sub">📅 Avg Tenure</div><div class="metric-value">{avg_tenure_fmt}</div></div>', unsafe_allow_html=True)
    with k4:
        avg_monthly = df2["MonthlyCharges"].mean() if "MonthlyCharges" in df2.columns and df2["MonthlyCharges"].dropna().shape[0] > 0 else None
        avg_monthly_fmt = "N/A" if avg_monthly is None or (isinstance(avg_monthly, float) and np.isnan(avg_monthly)) else f"{avg_monthly:.2f}"
        st.markdown(f'<div class="metric-card"><div class="metric-sub">💳 Avg Monthly</div><div class="metric-value">{avg_monthly_fmt}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- Charts --------------------
    left_col, right_col = st.columns([2, 1])

    # Left: retention + monthly box + churn by contract
    with left_col:
        st.markdown('<div class="chart-card"><strong>📈 Retention (Tenure Survival)</strong></div>', unsafe_allow_html=True)
        if tenure_col != "(none)" and df2[tenure_col].dropna().shape[0] > 0:
            total = len(df2)
            try:
                max_m = int(min(df2[tenure_col].dropna().astype(int).max(), 48))
            except Exception:
                try:
                    max_m = int(min(int(df2[tenure_col].dropna().max()), 48))
                except Exception:
                    max_m = 12
            months = list(range(0, max_m + 1))
            retention = [(df2[tenure_col] >= m).sum() / max(total, 1) for m in months]
            ret_df = pd.DataFrame({"month": months, "retention_rate": retention})
            fig = px.line(ret_df, x="month", y="retention_rate", markers=True, color_discrete_sequence=[ACCENT])
            fig.update_traces(line=dict(width=4), marker=dict(size=6))
            fig.update_yaxes(tickformat="%")
            fig = dark_plotly_layout(fig, height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Provide numeric tenure column to show retention curve.")

        st.markdown('<div class="chart-card" style="margin-top:12px;"><strong>📦 Monthly Charges by Churn (Box)</strong></div>', unsafe_allow_html=True)
        if churn_key and "MonthlyCharges" in df2.columns:
            tmp = df2[["MonthlyCharges", churn_key]].dropna()
            if tmp.shape[0] > 0:
                tmp[churn_key] = tmp[churn_key].astype(int).map({0: "No", 1: "Yes"})
                fig = px.box(tmp, x=churn_key, y="MonthlyCharges", points="all", color_discrete_sequence=PALETTE)
                fig = dark_plotly_layout(fig, height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rows to plot for MonthlyCharges by churn.")
        else:
            st.info("MonthlyCharges or Churn mapping missing to plot boxplot.")

        st.markdown('<div class="chart-card" style="margin-top:12px;"><strong>📊 Churn Rate by Contract Type</strong></div>', unsafe_allow_html=True)
        if "Contract" in df2.columns and churn_key:
            tmp = df2[["Contract", churn_key]].dropna()
            if tmp.shape[0] > 0:
                agg = tmp.groupby("Contract").agg(total=("Contract", "count"), churned=(churn_key, "sum")).reset_index()
                agg["churn_rate"] = agg["churned"] / agg["total"]
                agg = agg.sort_values("churn_rate", ascending=False)
                fig = px.bar(agg, x="Contract", y="churn_rate", text=agg["churn_rate"].apply(lambda x: "{:.0%}".format(x)), color_discrete_sequence=PALETTE)
                fig.update_traces(marker_line_width=0)
                fig = dark_plotly_layout(fig, height=360, rotate_x=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to compute churn by Contract.")
        else:
            st.info("Contract or Churn column missing.")

    # Right: churn distribution & service flags
    with right_col:
        st.markdown('<div class="chart-card"><strong>🍰 Churn Distribution</strong></div>', unsafe_allow_html=True)
        if churn_key:
            counts = df2[churn_key].dropna().astype(int).value_counts().sort_index()
            vals = [int(counts.get(0, 0)), int(counts.get(1, 0))]
            labels = ["Retained", "Churned"]
            fig = px.pie(values=vals, names=labels, hole=0.45, color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo="percent+label", textfont_size=13)
            fig = dark_plotly_layout(fig, height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Map the Churn column to view distribution.")

        st.markdown('<div class="chart-card" style="margin-top:12px;"><strong>🔍 Top Drivers (Service Flags)</strong></div>', unsafe_allow_html=True)
        service_cols = [c for c in ["TechSupport", "OnlineSecurity", "DeviceProtection", "OnlineBackup", "StreamingTV", "StreamingMovies", "PhoneService"] if c in df2.columns]
        if service_cols and churn_key:
            rows = []
            for c in service_cols:
                tmp = df2[[c, churn_key]].dropna()
                if tmp.shape[0] == 0:
                    continue
                mapped = tmp[c].astype(str).str.strip().str.lower().replace({"yes": 1, "no": 0, "no phone service": 0})
                tmp_local = tmp.copy()
                tmp_local["_flag"] = mapped
                tmp_local = tmp_local.dropna(subset=["_flag"])
                if tmp_local.shape[0] == 0:
                    continue
                agg = tmp_local.groupby("_flag").agg(total=("_flag", "count"), churned=(churn_key, "sum")).reset_index()
                if 1 in agg["_flag"].values:
                    try:
                        churn_rate_flag = float(agg.loc[agg["_flag"] == 1, "churned"].values[0]) / float(agg.loc[agg["_flag"] == 1, "total"].values[0])
                    except Exception:
                        churn_rate_flag = 0.0
                else:
                    churn_rate_flag = 0.0
                rows.append({"feature": c, "churn_rate_if_yes": churn_rate_flag})
            feat_df = pd.DataFrame(rows).sort_values("churn_rate_if_yes", ascending=False)
            if not feat_df.empty:
                fig = px.bar(feat_df, x="churn_rate_if_yes", y="feature", orientation="h", text=feat_df["churn_rate_if_yes"].apply(lambda x: "{:.0%}".format(x)), color_discrete_sequence=PALETTE)
                fig = dark_plotly_layout(fig, height=360, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No usable service flags found.")
        else:
            st.info("Service flag columns or churn mapping missing.")

    st.markdown("---")

    # Payment method & internet service
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card"><strong>💳 Churn by Payment Method</strong></div>', unsafe_allow_html=True)
        if "PaymentMethod" in df2.columns and churn_key:
            tmp = df2[["PaymentMethod", churn_key]].dropna()
            if tmp.shape[0] > 0:
                agg = tmp.groupby("PaymentMethod").agg(total=("PaymentMethod", "count"), churned=(churn_key, "sum")).reset_index()
                agg["churn_rate"] = agg["churned"] / agg["total"]
                agg = agg.sort_values("churn_rate", ascending=False)
                fig = px.bar(agg, x="PaymentMethod", y="churn_rate", text=agg["churn_rate"].apply(lambda x: "{:.0%}".format(x)), color_discrete_sequence=PALETTE)
                fig = dark_plotly_layout(fig, height=350, rotate_x=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough rows to compute PaymentMethod churn.")
        else:
            st.info("PaymentMethod or Churn missing.")

    with c2:
        st.markdown('<div class="chart-card"><strong>🌐 Churn by Internet Service</strong></div>', unsafe_allow_html=True)
        if "InternetService" in df2.columns and churn_key:
            tmp = df2[["InternetService", churn_key]].dropna()
            if tmp.shape[0] > 0:
                agg = tmp.groupby("InternetService").agg(total=("InternetService", "count"), churned=(churn_key, "sum")).reset_index()
                agg["churn_rate"] = agg["churned"] / agg["total"]
                agg = agg.sort_values("churn_rate", ascending=False)
                fig = px.bar(agg, x="InternetService", y="churn_rate", text=agg["churn_rate"].apply(lambda x: "{:.0%}".format(x)), color_discrete_sequence=PALETTE)
                fig = dark_plotly_layout(fig, height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough rows to compute InternetService churn.")
        else:
            st.info("InternetService or Churn missing.")

    st.markdown("---")

    # Correlation heatmap
    st.markdown('<div class="chart-card"><strong>🧭 Numeric Correlation</strong></div>', unsafe_allow_html=True)
    num_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df2.columns]
    if num_cols and churn_key:
        corr_df = df2[num_cols].copy()
        for c in corr_df.columns:
            corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
        corr_df["_churn_for_corr"] = pd.to_numeric(df2[churn_key], errors="coerce")
        corr_df = corr_df.dropna(how="all")
        if corr_df.shape[0] >= 2:
            corr = corr_df.corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdYlBu", reversescale=True))
            fig.update_traces(colorbar=dict(title="corr"))
            fig = dark_plotly_layout(fig, height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric rows after coercion to compute correlation.")
    else:
        st.info("Not enough numeric columns for correlation.")

    st.markdown("---")

    # Export filtered dataset
    try:
        csv_all = df2.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download Filtered Dataset", data=csv_all, file_name="churn_analysis_filtered.csv")
    except Exception:
        st.info("Could not prepare download file.")

    # Auto summary (quick)
    try:
        tot = total_customers
        churn_count = int(df2[churn_key].dropna().astype(int).sum()) if churn_key else None
        churn_pct = (churn_count / tot) if (churn_count is not None and tot > 0) else None
        avg_t = avg_tenure_fmt
        avg_m = avg_monthly_fmt
        summary_html = '<div class="info-card"><strong>🔎 Quick Summary</strong><br>'
        summary_html += f'<div style="margin-top:8px;">Total customers: {tot}</div>'
        summary_html += f'<div>Overall churn rate: {("{:.1%}".format(churn_pct) if churn_pct is not None else "N/A")}</div>'
        summary_html += f'<div>Avg tenure: {avg_t}</div>'
        summary_html += f'<div>Avg monthly charge: {avg_m}</div>'
        summary_html += '</div>'
        st.markdown(summary_html, unsafe_allow_html=True)
    except Exception:
        pass

# ------------------ RAG UI (fast, safe, single block) --------------------
st.markdown("---")
st.markdown('<div class="chart-card"><strong>🔎 RAG Q&A (ask about your dataset)</strong></div>', unsafe_allow_html=True)

# choose source df (prefer filtered in session_state)
_df_for_rag = st.session_state.get("filtered_df", df)

# create a small signature to know when to rebuild
_source_sig = (tuple(_df_for_rag.columns), len(_df_for_rag))

if st.session_state.get("_rag_source_sig") != _source_sig:
    st.session_state["_rag_source_sig"] = _source_sig
    st.session_state["rag_passages"] = build_corpus_from_df_sampled(_df_for_rag, max_rows=MAX_RAG_ROWS, max_chars=MAX_PASSAGE_CHARS)
    st.session_state["rag_vect"], st.session_state["rag_mat"] = build_tfidf_index_safe(st.session_state.get("rag_passages", []), max_features=TFIDF_MAX_FEATURES)

if "rag_passages" not in st.session_state:
    st.session_state["rag_passages"] = []
if "rag_vect" not in st.session_state:
    st.session_state["rag_vect"] = None
if "rag_mat" not in st.session_state:
    st.session_state["rag_mat"] = None

st.caption(f"Indexing: up to {MAX_RAG_ROWS} rows sampled • truncating long rows • TF-IDF max features: {TFIDF_MAX_FEATURES}")

user_question = st.text_input("Ask a question about the dataset (e.g., 'Which contract has highest churn?')", key="rag_ui_question")
top_k = st.number_input("Top-K passages", min_value=1, max_value=10, value=3, key="rag_ui_topk")
gemini_checkbox = st.checkbox("Use Gemini (if configured in secrets) — WARNING: may block/wait", value=False, key="rag_ui_gemini")

if st.button("Run RAG", key="run_rag_ui"):
    if not user_question or not user_question.strip():
        st.warning("Please enter a question.")
    elif st.session_state["rag_vect"] is None or st.session_state["rag_mat"] is None or len(st.session_state["rag_passages"]) == 0:
        st.error("Retriever not ready or no passages indexed. Try re-running analysis or reduce MAX_RAG_ROWS.")
    else:
        try:
            with st.spinner("Retrieving relevant passages..."):
                retrieved = retrieve_top_k_from_index(user_question, st.session_state["rag_vect"], st.session_state["rag_mat"], st.session_state["rag_passages"], k=top_k)
            if not retrieved:
                st.info("No relevant passages found.")
            else:
                st.success(f"Found {len(retrieved)} relevant passages.")
                context_text = ""
                for score, passage, idx in retrieved:
                    st.markdown(f"**Score:** {score:.3f}")
                    st.markdown(textwrap.fill(passage, width=140))
                    st.markdown("---")
                    context_text += passage + "\n\n"

                gemini_answer = None
                if gemini_checkbox:
                    try:
                        st.info("Calling Gemini (this may take several seconds)...")
                        gemini_answer = call_gemini_generate(context_text, user_question)
                    except Exception as e:
                        st.error(f"Gemini call failed: {e}")
                        gemini_answer = None

                if gemini_answer:
                    st.subheader("Answer from Gemini")
                    st.write(gemini_answer)
                else:
                    st.subheader("Extractive fallback answer")
                    summary = extractive_summary_from_passages([p for _, p, _ in retrieved], user_question, max_sentences=4)
                    if summary:
                        st.write(summary)
                    else:
                        st.info("No extractive sentences found; see retrieved passages above.")
        except Exception as err:
            st.error("RAG pipeline failed — see details below.")
            st.exception(err)
