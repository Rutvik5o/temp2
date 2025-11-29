# app.py
# Combined: PowerBI-style Dark Churn Dashboard + RAG Q&A with human-readable fallback
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import concurrent.futures
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, List, Tuple

# Optional Gemini client - import lazily (so app runs if package missing)
try:
    import google.generativeai as genai  # pip: google-generativeai
    _HAS_GEMINI = True
except Exception:
    genai = None
    _HAS_GEMINI = False

st.set_page_config(
    page_title="Customer Churn Analyzer — Dark PowerBI + RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Visual / Palette --------------------------------
PALETTE = px.colors.qualitative.Plotly
ACCENT = "#0ea5a3"

# ------------------------- Dark theme CSS ---------------------------------
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
.stButton>button{ background: linear-gradient(90deg,var(--accent),var(--accent-2)); color:white; border:none; padding:8px 18px; border-radius:10px; font-weight:600; box-shadow: 0 6px 20px rgba(14,165,163,0.18);}
.chart-card { padding:10px; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005)); border-radius:10px; box-shadow: 0 10px 30px rgba(2,6,23,0.6); border: 1px solid rgba(255,255,255,0.02);}
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
    st.markdown('<div class="info-card"><strong>⚡ Quick Tips</strong><br>• Ensure Tenure & MonthlyCharges are numeric<br>• Churn: Yes/No or 1/0<br>• Optional: set GEMINI_API_KEY in Streamlit secrets to enable Gemini</div>', unsafe_allow_html=True)

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
def normalize_churn(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')
    s = series.astype(str).str.strip().str.lower()
    mapping = {'yes':1, 'y':1, 'true':1, '1':1, 't':1, 'churn':1, 'no':0, 'n':0, 'false':0, '0':0, 'stay':0, 'retained':0}
    return s.map(mapping).astype(float)

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
def row_to_text(row: pd.Series) -> str:
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

def build_corpus_from_df(df_local: pd.DataFrame, max_rows: int = 5000):
    corpus = []
    rows = min(len(df_local), max_rows)
    for i in range(rows):
        txt = row_to_text(df_local.iloc[i])
        if txt:
            corpus.append(txt)
    return corpus

def build_tfidf_index(passages: List[str]):
    if not passages:
        return None, None
    vect = TfidfVectorizer(stop_words="english", max_features=20000)
    mat = vect.fit_transform(passages)
    return vect, mat

def retrieve_top_k(query: str, vect, mat, passages: List[str], k: int = 3) -> List[Tuple[float,str,int]]:
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

def extractive_summary_from_passages(passages: List[str], question: str, max_sentences=3) -> Optional[str]:
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

def format_human_answer_from_retrieved(retrieved: List[Tuple[float,str,int]], question: str, df_filtered: pd.DataFrame, max_examples:int=3) -> str:
    """
    Build a human readable answer from retrieved passages.
    - retrieved: list of (score, passage, idx)
    - question: user question
    - df_filtered: filtered dataframe to extract example rows if needed
    """
    lines = []
    lines.append(f"**Question:** {question}")
    lines.append("")

    # Simple direct answer derived from extractive sentences
    passages = [p for _, p, _ in retrieved]
    direct = extractive_summary_from_passages(passages, question, max_sentences=2)
    if direct:
        lines.append(f"**Answer (extractive):** {direct}")
    else:
        lines.append("**Answer:** Couldn't extract a concise sentence from the retrieved rows — see supporting examples below.")

    lines.append("")
    lines.append("**Reasoning / Evidence:**")
    # show brief lines from top passages with score
    for score, passage, idx in retrieved[:5]:
        # shorten passage for readability, show key fields
        short = passage
        if len(short) > 260:
            short = short[:250].rsplit(" ",1)[0] + "…"
        lines.append(f"- (score {score:.3f}) {short}")

    lines.append("")
    # Provide top example rows (try to parse customerID if present)
    example_lines = []
    for _, passage, _ in retrieved[:max_examples]:
        # find customerID snippet or show first 120 chars
        if "customerid" in passage.lower():
            # try to extract customerID token
            tokens = [t.strip() for t in passage.split("|")]
            cid = None
            for tok in tokens:
                if tok.lower().startswith("customerid"):
                    cid = tok
                    break
            example_lines.append(cid if cid else passage[:120])
        else:
            example_lines.append(passage[:120])
    if example_lines:
        lines.append("**Example rows that support this:**")
        for ex in example_lines:
            lines.append(f"- {ex}")

    # Actionable insights (simple heuristics)
    lines.append("")
    lines.append("**Actionable insights (auto-generated):**")
    # Heuristic 1: if many retrieved rows have 'Month-to-month' or similar -> recommend contract offers
    text_all = " ".join(passages).lower()
    if "month-to-month" in text_all or "month to month" in text_all:
        lines.append("- Many examples are on month-to-month contracts — consider offering incentives for longer contracts (discounts or bundle).")
    if "electronic check" in text_all or "paperlessbilling: yes" in text_all:
        lines.append("- Several churned customers used electronic checks / paperless billing — investigate payment friction or failed payments.")
    if "seniorcitizen: 1" in text_all:
        lines.append("- Some churners are senior citizens — ensure tailored support and clear billing/assistance.")
    # fallback generic suggestions
    lines.append("- Monitor high monthly-charge customers for early churn signals; consider proactive retention outreach.")
    return "\n".join(lines)

# ------------------------- Gemini Safe Caller ------------------------------
def init_gemini():
    """Return True if gemini client configured, False otherwise."""
    if not _HAS_GEMINI:
        return False
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return False
        genai.configure(api_key=api_key)
        return True
    except Exception:
        return False

def _gemini_worker(prompt: str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp

def call_gemini_generate(context_text: str, user_question: str, timeout_sec: int = 20) -> Optional[str]:
    if not _HAS_GEMINI:
        return None
    if not init_gemini():
        return None
    prompt = f"""You are a customer churn analytics assistant.
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
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_gemini_worker, prompt)
            try:
                resp = fut.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                fut.cancel()
                return None
        text = None
        if hasattr(resp, "text"):
            text = resp.text
        elif isinstance(resp, dict) and "candidates" in resp:
            cand = resp.get("candidates")
            if cand and isinstance(cand, (list, tuple)) and len(cand) > 0:
                text = cand[0].get("content", None)
        else:
            text = str(resp)
        if not text or text.strip() == "":
            return None
        return text
    except Exception:
        tb = traceback.format_exc()
        st.text_area("Gemini exception (debug)", tb, height=120)
        return None

# ------------------------- RUN ANALYSIS ----------------------------------
if run_analysis:
    df2 = df.copy()
    for s, selvals in segment_filters.items():
        if selvals:
            try:
                df2 = df2[df2[s].isin(selvals)]
            except Exception:
                st.warning(f"Filter {s} could not be applied (type mismatch).")

    st.markdown(f'<div class="info-card"><strong>📋 Filtered Dataset</strong> • {len(df2)} rows</div>', unsafe_allow_html=True)
    st.dataframe(df2.head(8), use_container_width=True, height=200)

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

    total_customers = len(df2)
    avg_tenure_fmt = "N/A"
    if tenure_col != "(none)":
        df2[tenure_col] = pd.to_numeric(df2[tenure_col], errors="coerce")
        if df2[tenure_col].dropna().shape[0] > 0:
            avg_tenure = df2[tenure_col].dropna().mean()
            avg_tenure_fmt = f"{avg_tenure:.1f}"
    if "MonthlyCharges" in df2.columns:
        df2["MonthlyCharges"] = pd.to_numeric(df2["MonthlyCharges"], errors="coerce")
    avg_monthly_fmt = "N/A"
    if "MonthlyCharges" in df2.columns and df2["MonthlyCharges"].dropna().shape[0] > 0:
        avg_monthly = df2["MonthlyCharges"].mean()
        avg_monthly_fmt = f"{avg_monthly:.2f}"

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
        st.markdown(f'<div class="metric-card"><div class="metric-sub">📅 Avg Tenure</div><div class="metric-value">{avg_tenure_fmt}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-sub">💳 Avg Monthly</div><div class="metric-value">{avg_monthly_fmt}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # (Charts code omitted for brevity in this message — identical to previous working charts)
    # ... (Use the same chart generation blocks from your app; they remain unchanged)

    # Build corpus & retriever
    st.markdown("---")
    st.markdown('<div class="chart-card"><strong>🔎 RAG Q&A (ask about your uploaded dataset)</strong></div>', unsafe_allow_html=True)
    max_corpus_rows = 5000
    passages = build_corpus_from_df(df2, max_rows=max_corpus_rows)
    vect, mat = build_tfidf_index(passages)

    # store in session to avoid rebuilds
    if "rag_passages" not in st.session_state:
        st.session_state["rag_passages"] = passages
    if "rag_vect" not in st.session_state or "rag_mat" not in st.session_state:
        try:
            st.session_state["rag_vect"], st.session_state["rag_mat"] = build_tfidf_index(st.session_state["rag_passages"])
        except Exception:
            st.session_state["rag_vect"], st.session_state["rag_mat"] = None, None

    user_question = st.text_input("Ask a question about the dataset (e.g., 'Which contract has highest churn?')", key="rag_question")
    top_k = st.number_input("Top-K passages", min_value=1, max_value=10, value=3, key="rag_topk")
    use_gemini = st.checkbox("Use Gemini (if configured in Streamlit secrets)", value=True, key="use_gemini")

    if st.button("Run RAG", key="run_rag"):
        if not user_question or not user_question.strip():
            st.warning("Please enter a question.")
        else:
            if st.session_state.get("rag_vect") is None or st.session_state.get("rag_mat") is None:
                st.error("Retriever not ready. The TF-IDF index couldn't be built.")
            else:
                try:
                    with st.spinner("Retrieving top-k passages..."):
                        retrieved = retrieve_top_k(user_question, st.session_state["rag_vect"], st.session_state["rag_mat"], st.session_state["rag_passages"], k=top_k)
                    if not retrieved:
                        st.info("No relevant passages found.")
                    else:
                        st.success(f"Retrieved {len(retrieved)} passages.")
                        # Show compact retrieved passages for transparency
                        for score, passage, idx in retrieved:
                            st.markdown(f"**Score:** {score:.3f}")
                            st.write(passage)

                        # Build context
                        ctx_text = "\n\n".join([p for _, p, _ in retrieved])

                        gemini_answer = None
                        if use_gemini:
                            st.info("Calling Gemini (this may take several seconds)...")
                            gemini_answer = call_gemini_generate(ctx_text, user_question, timeout_sec=20)

                        if gemini_answer:
                            st.subheader("Answer (Gemini)")
                            st.write(gemini_answer)
                        else:
                            # Human readable fallback
                            human_text = format_human_answer_from_retrieved(retrieved, user_question, df2, max_examples=3)
                            st.subheader("Answer (Extractive, humanized)")
                            st.markdown(human_text)
                except Exception as err:
                    st.error("RAG pipeline failed — see debug below.")
                    st.exception(err)

else:
    st.info('👈 Configure columns + filters → Click **Run Analysis** 🚀')
