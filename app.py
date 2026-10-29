"""NLP Module Inspector — Streamlit debugging UI for the Vietnamese stock sentiment model."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from nlp_module import volcast
from nlp_module.text import normalize
from nlp_module.sentiment.base import LABELS, entropy as calc_entropy

# ── constants ───────────────────────────────────────────────────────────────
# NOT "Tăng"/"Giảm": the model judges the company's prospects in the passage, never the price direction
LABEL_VI = {"positive": "Tích cực", "neutral": "Trung tính", "negative": "Tiêu cực"}
# diverging: blue (positive) <-> gray <-> red (negative); green/red is the classic colour-blind failure
LABEL_COLOR = {"positive": "#2a78d6", "neutral": "#898781", "negative": "#e34948"}
DIVERGING = [[0, "#e34948"], [0.5, "#d8d7d1"], [1, "#2a78d6"]]   # sentiment score -1 .. +1
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NLP Inspector",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.stTabs [data-baseweb="tab"] { padding: 0 22px; }
</style>""", unsafe_allow_html=True)


# ── helpers ─────────────────────────────────────────────────────────────────
def vi(l): return LABEL_VI.get(l, l)

def token_html(tokens, scores, label="positive"):
    if not tokens:
        return "<em style='color:gray'>No token data</em>"
    mx = max(abs(s) for s in scores) or 1e-9
    rgb = {"positive": "42,120,214", "neutral": "137,135,129", "negative": "227,73,72"}
    c = rgb.get(label, "137,135,129")
    parts = []
    for tok, sc in zip(tokens, scores):
        a = 0.12 + 0.75 * min(abs(sc) / mx, 1.0)
        tok_disp = tok.replace("_", "&#x5f;").replace("<", "&lt;")
        parts.append(
            f'<span style="background:rgba({c},{a:.2f});border-radius:3px;'
            f'padding:2px 6px;margin:2px;display:inline-block;font-size:.88em;'
            f'line-height:1.9">{tok_disp}</span>'
        )
    return " ".join(parts)


def prob_chart(probs):
    fig = go.Figure(go.Bar(
        x=[vi(l) for l in LABELS], y=list(probs),
        marker_color=[LABEL_COLOR[l] for l in LABELS],
        text=[f"{p:.1%}" for p in probs], textposition="outside",
    ))
    fig.update_layout(yaxis=dict(range=[0, 1.15], title="Xác suất"),
                      height=245, margin=dict(t=16, b=16, l=20, r=20),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def confusion_fig(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    lvi = [vi(l) for l in LABELS]
    fig = px.imshow(cm, x=lvi, y=lvi, labels={"x": "Dự đoán", "y": "Thật"},
                    color_continuous_scale="Blues", text_auto=True)
    fig.update_layout(height=320, margin=dict(t=40, b=40, l=60, r=20),
                      coloraxis_showscale=False)
    return fig


def reliability_fig(y_true, all_probs):
    arr = np.array(all_probs)
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="rgba(128,128,128,.4)"))
    bins = np.linspace(0, 1, 11)
    for li, label in enumerate(LABELS):
        bx, by, bn = [], [], []
        for i in range(10):
            lo, hi = bins[i], bins[i + 1]
            mask = (arr[:, li] >= lo) & (arr[:, li] < hi)
            if not mask.any():
                continue
            bx.append(float(arr[mask, li].mean()))
            by.append(float((np.array(y_true)[mask] == label).mean()))
            bn.append(int(mask.sum()))
        fig.add_trace(go.Scatter(x=bx, y=by, mode="markers+lines", name=vi(label),
                                  marker=dict(size=[max(6, n // 4) for n in bn],
                                              color=LABEL_COLOR[label], opacity=.85)))
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1], height=320,
                      xaxis_title="Độ tự tin", yaxis_title="Accuracy",
                      margin=dict(t=16, b=40, l=55, r=16),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── data ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_DIR / "eval_labeled.csv", sep=";")
    df["source"] = df["url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0].fillna("unknown")
    df["text_len"] = df["passage_text"].fillna("").str.len()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    # flag rows with non-standard labels (skip / unclear) — keep them visible but exclude from metrics
    df["valid_label"] = df["label"].isin(LABELS)
    return df


SNAP_FILE = {"baseline-tfidf-lr-v1": "snapshots_baseline_backtest.parquet",
             "phobert-phobert-v1": "snapshots_phobert_backtest.parquet"}


@st.cache_data(ttl=300)
def db_versions() -> list:
    import sqlite3
    con = sqlite3.connect(f"file:{DATA_DIR / 'nlp.db'}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute("SELECT nlp_model_version, COUNT(*) n FROM article_company_result "
                                          "GROUP BY 1 ORDER BY 2 DESC, 1")]
    finally:
        con.close()


@st.cache_data(ttl=300)
def db_results(version: str) -> pd.DataFrame:
    """Every scored (article, company) row of one model version, deduplicated to representative versions."""
    import sqlite3
    con = sqlite3.connect(f"file:{DATA_DIR / 'nlp.db'}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("""
            SELECT r.symbol, v.published_at, v.title, d.url_normalized AS url, d.source,
                   r.relation, r.matching_score, r.passage_text,
                   r.p_pos, r.p_neu, r.p_neg, r.entropy, r.uncertain, r.difficult, r.needs_review
            FROM article_company_result r
            JOIN document_version v USING(document_version_id)
            JOIN document d USING(document_id)
            WHERE r.nlp_model_version = ? AND v.is_representative = 1 AND v.published_at IS NOT NULL
        """, con, params=(version,))
    finally:
        con.close()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["pred"] = [LABELS[i] for i in df[["p_pos", "p_neu", "p_neg"]].to_numpy().argmax(1)]
    df["score"] = df["p_pos"] - df["p_neg"]          # the same quantity that feeds sentiment_mean
    for c in ("uncertain", "difficult", "needs_review"):
        df[c] = df[c].astype(bool)
    return df.sort_values("published_at", ascending=False)


@st.cache_data(ttl=300)
def db_quotes() -> pd.DataFrame:
    """Every stored SSI board reading. Read-only: fetching new ones is an explicit button press."""
    import sqlite3
    con = sqlite3.connect(f"file:{DATA_DIR / 'nlp.db'}?mode=ro", uri=True)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='market_quote'").fetchone():
            return pd.DataFrame()
        df = pd.read_sql_query("SELECT * FROM market_quote", con)
    finally:
        con.close()
    if df.empty:
        return df
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    return df.sort_values("fetched_at")


@st.cache_data(ttl=300)
def db_vol_forecast(version: str) -> pd.DataFrame:
    """Newest vol_forecast row per (symbol, h) for one model version. Read-only; empty if table/rows missing."""
    import sqlite3
    con = sqlite3.connect(f"file:{DATA_DIR / 'nlp.db'}?mode=ro", uri=True)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE name='vol_forecast'").fetchone():
            return pd.DataFrame()
        return pd.read_sql_query(
            "SELECT symbol, base_date, h, p_strong, call FROM vol_forecast WHERE model_version=? "
            "AND base_date=(SELECT MAX(base_date) FROM vol_forecast v WHERE v.symbol=vol_forecast.symbol "
            "AND v.h=vol_forecast.h AND v.model_version=vol_forecast.model_version)", con, params=(version,))
    finally:
        con.close()


@st.cache_data(ttl=300)
def vol_report() -> dict:
    import json
    f = ROOT / volcast.REPORT
    return json.loads(f.read_text()) if f.exists() else {}


def fetch_quotes_now(symbols, version):
    """One polite request to the SSI board + the live sentiment join. Only ever called from a button."""
    from nlp_module import storage
    from nlp_module.connectors import ssi
    from nlp_module.http import Fetcher
    from nlp_module.pipeline import sentiment_vs_price
    c = storage.connect()
    qs = ssi.quotes(Fetcher(min_interval=1.0), symbols)
    storage.save_quotes(c, qs, ssi.SOURCE)
    rows = sentiment_vs_price(c, qs, version)
    c.close()
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def latest_cmp(version, symbols):
    """The last stored board reading (cron or button) joined with live sentiment - no request to SSI."""
    from nlp_module import storage
    from nlp_module.connectors.ssi import Quote
    from nlp_module.pipeline import sentiment_vs_price
    q = db_quotes()
    if q.empty:
        return pd.DataFrame(), None
    run = q[q["fetched_at"] >= q["fetched_at"].max() - pd.Timedelta(minutes=2)]   # one run = one board read
    run = run[run["symbol"].isin(symbols)].groupby("symbol").tail(1)
    qs = [Quote(**{k: (bool(r[k]) if k == "matched" else r[k]) for k in Quote.__dataclass_fields__})
          for r in run.to_dict("records")]
    c = storage.connect()
    rows = sentiment_vs_price(c, qs, version)
    c.close()
    return pd.DataFrame(rows), q["fetched_at"].max()


@st.cache_data
def load_prices():
    f = DATA_DIR / "prices.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "symbol", "close"])
    return pd.read_csv(f, parse_dates=["date"])


@st.cache_data
def db_snapshots(version: str):
    f = DATA_DIR / SNAP_FILE.get(version, "__none__")
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d["forecast_cutoff"] = pd.to_datetime(d["forecast_cutoff"], utc=True)
    return d


# ── model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(key: str):
    if key.endswith(".joblib"):
        import joblib
        obj = joblib.load(MODEL_DIR / key)
        pipe, ver = obj if isinstance(obj, tuple) else (obj, key)
        return {"type": "baseline", "model": pipe, "version": ver}
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = MODEL_DIR / key
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="eager").eval()  # sdpa không trả attentions
    return {"type": "phobert", "model": mdl, "tok": tok, "version": key}


def _segment(text: str) -> str:
    try:
        from nlp_module.sentiment.segment import segment
        return segment([text])[0]
    except Exception:
        return text  # fallback: raw text (VnCoreNLP/Java unavailable)


def predict_single(m, text: str):
    """Returns (probs, logits, explain)."""
    if m["type"] == "baseline":
        pipe = m["model"]
        proba = pipe.predict_proba([text])[0]
        cls = list(pipe.classes_)
        probs = tuple(float(proba[cls.index(l)]) if l in cls else 0.0 for l in LABELS)
        try:
            logits = list(map(float, pipe.decision_function([text])[0]))
        except Exception:
            logits = []
        explain = {"type": "baseline", "contribs": {}}
        try:
            vec = pipe.named_steps["tfidfvectorizer"]
            lr = pipe.named_steps["logisticregression"]
            X = vec.transform([text])
            fn = vec.get_feature_names_out()
            nz = X.nonzero()[1]
            tfidf_vals = np.asarray(X[0, nz]).flatten()
            lr_cls = list(lr.classes_)
            for label in LABELS:
                li = lr_cls.index(label) if label in lr_cls else 0
                items = sorted([(fn[i], float(v * lr.coef_[li, i]))
                                 for i, v in zip(nz, tfidf_vals)],
                               key=lambda x: abs(x[1]), reverse=True)[:20]
                explain["contribs"][label] = items
        except Exception:
            pass
        return probs, logits, explain

    else:
        import torch
        seg = _segment(text)
        tok, mdl = m["tok"], m["model"]
        enc = tok([seg], truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**enc, output_attentions=True)
        probs = tuple(float(p) for p in torch.softmax(out.logits[0], -1).tolist())
        logits = list(map(float, out.logits[0].tolist()))
        toks = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        last_attn = out.attentions[-1][0].mean(0)[0]  # CLS row, avg heads
        inner = toks[1:-1]
        attn = last_attn[1:len(toks) - 1].tolist()
        explain = {"type": "phobert", "tokens": inner, "attn": attn,
                   "truncated": len(toks) >= 256,
                   "unk_count": sum(1 for t in inner if "[unk]" in t.lower())}
        return probs, logits, explain


@st.cache_data(show_spinner=False)
def batch_predict(model_key: str, texts: tuple) -> list:
    m = load_model(model_key)
    text_list = list(texts)
    if m["type"] == "baseline":
        pipe = m["model"]
        proba = pipe.predict_proba(text_list)
        cls = list(pipe.classes_)
        return [tuple(float(row[cls.index(l)]) if l in cls else 0.0 for l in LABELS)
                for row in proba]
    import torch
    segs = [_segment(t) for t in text_list]
    tok, mdl = m["tok"], m["model"]
    out = []
    for i in range(0, len(segs), 16):
        enc = tok(segs[i:i + 16], truncation=True, max_length=256, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits.float()
        out.extend(tuple(float(x) for x in row)
                   for row in torch.softmax(logits, -1).numpy())
    return out


def get_preds():
    if st.session_state.get("preds_key") == model_key:
        return st.session_state.get("preds_data")
    return None


# ── sidebar ──────────────────────────────────────────────────────────────────
df_all = load_dataset()
with st.sidebar:
    st.title("NLP Inspector")
    joblibs = sorted(p.name for p in MODEL_DIR.iterdir() if p.suffix == ".joblib")
    phobts = sorted(p.name for p in MODEL_DIR.iterdir() if p.is_dir())
    model_key = st.selectbox("Model", joblibs + phobts)

    st.divider()
    n = len(df_all)
    if st.button(f"Nạp predictions ({n} mẫu)", width="stretch"):
        with st.spinner(f"Inference {model_key} …"):
            preds = batch_predict(model_key, tuple(df_all["passage_text"].tolist()))
            st.session_state["preds_key"] = model_key
            st.session_state["preds_data"] = preds
        st.success(f"Đã nạp {len(preds)} mẫu")

    st.divider()
    st.caption(f"{n} mẫu · {df_all['symbol'].nunique()} mã CK")


# ── tabs ─────────────────────────────────────────────────────────────────────
tab_stock, tab_price, tab_vol, tab_3d, tab_play, tab_explore, tab_whatif, tab_calib = st.tabs(
    ["Theo mã CK", "Sentiment vs Giá SSI", "Dự báo biến động", "Toàn cảnh 3D", "Playground",
     "Dataset Explorer", "What-if", "Calibration"])


# ════════════════════════════════════════════════════════════════════════════
# PLAYGROUND
# ════════════════════════════════════════════════════════════════════════════
with tab_play:
    st.header("Single-sample Inspector")

    src = st.radio("Nguồn", ["Nhập tay", "Chọn từ dataset"], horizontal=True)
    true_label = None

    if src == "Chọn từ dataset":
        idx = st.selectbox("Mẫu", range(len(df_all)),
                           format_func=lambda i: f"[{df_all.iloc[i]['symbol']}] "
                                                 f"{str(df_all.iloc[i]['title'])[:80]}…")
        row = df_all.iloc[idx]
        text_input = str(row["passage_text"])
        true_label = row["label"]

        with st.expander("Metadata", expanded=False):
            mc = st.columns(4)
            mc[0].write(f"**Mã CK:** {row['symbol']}")
            mc[1].write(f"**Ngày:** {str(row['published_at'])[:10]}")
            mc[2].write(f"**Nguồn:** {row['source']}")
            mc[3].write(f"**Khó:** {'Có' if row['difficult'] else 'Không'}")
            if row.get("note"):
                st.caption(f"Ghi chú: {row['note']}")
    else:
        text_input = st.text_area("Văn bản", height=120,
                                   placeholder="Dán tin tức, tiêu đề hoặc bình luận…")

    if st.button("Dự đoán", type="primary", disabled=not bool(text_input)):
        with st.spinner("Đang chạy…"):
            m = load_model(model_key)
            probs, logits, explain = predict_single(m, text_input)

        ent = calc_entropy(probs)
        pred = LABELS[int(np.argmax(probs))]

        c1, c2, c3, c4 = st.columns(4)
        pred_str = vi(pred) + (" (đúng)" if true_label and pred == true_label
                               else " (sai)" if true_label else "")
        c1.metric("Dự đoán", pred_str)
        if true_label:
            c2.metric("Nhãn thật", vi(true_label))
        else:
            c2.metric("Max prob", f"{max(probs):.1%}")
        c3.metric("Entropy", f"{ent:.3f}")
        c4.metric("Độ tự tin", f"{max(probs):.1%}")

        col_chart, col_log = st.columns([3, 1])
        with col_chart:
            st.plotly_chart(prob_chart(probs), width="stretch", key="play_probs")
        with col_log:
            st.caption("Logits")
            for l, v in zip(LABELS, logits if logits else [0.0] * 3):
                st.write(f"{vi(l)}: `{v:.4f}`")

        st.divider()

        # Preprocessing pipeline
        st.subheader("Pipeline tiền xử lý")
        norm = normalize(text_input)
        p1, p2 = st.columns(2)
        p1.caption("1. Gốc")
        p1.text_area("", text_input, height=90, disabled=True, label_visibility="collapsed")
        p2.caption("2. Chuẩn hóa (NFC + whitespace)")
        p2.text_area("", norm, height=90, disabled=True, label_visibility="collapsed")

        # Explainability
        st.subheader("Đóng góp token")
        if explain["type"] == "phobert":
            if explain.get("truncated"):
                st.warning("Văn bản bị truncate (≥ 256 tokens)")
            if explain.get("unk_count", 0):
                st.warning(f"{explain['unk_count']} token [UNK]")
            view_lbl = st.selectbox("Màu theo nhãn", LABELS, format_func=vi)
            toks, attn = explain.get("tokens", []), explain.get("attn", [])
            if toks:
                st.caption(f"Attention [CLS] → tokens (lớp cuối, avg heads) — {len(toks)} tokens")
                st.markdown(token_html(toks, attn, view_lbl), unsafe_allow_html=True)
        else:
            tabs_lbl = st.tabs([vi(l) for l in LABELS])
            for t, label in zip(tabs_lbl, LABELS):
                with t:
                    rows = explain.get("contribs", {}).get(label, [])
                    if rows:
                        df_c = pd.DataFrame(rows, columns=["token", "đóng góp"])
                        pos = df_c[df_c["đóng góp"] > 0].head(10)
                        neg = df_c[df_c["đóng góp"] < 0].head(10)
                        ca, cb = st.columns(2)
                        ca.caption(f"Đẩy → {vi(label)}")
                        ca.dataframe(pos, width="stretch", hide_index=True)
                        cb.caption(f"Kéo khỏi {vi(label)}")
                        cb.dataframe(neg, width="stretch", hide_index=True)
                    else:
                        st.info("Không có đóng góp.")


# ════════════════════════════════════════════════════════════════════════════
# DATASET EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.header("Dataset Explorer")
    all_probs = get_preds()
    has_preds = all_probs is not None

    if not has_preds:
        st.info("Bấm 'Nạp predictions' ở sidebar để mở confusion matrix và metrics.")

    wdf = df_all.copy()
    if has_preds:
        arr = np.array(all_probs)
        wdf["pred"] = [LABELS[int(np.argmax(p))] for p in all_probs]
        wdf["confidence"] = arr.max(axis=1)
        wdf["entropy"] = [calc_entropy(p) for p in all_probs]
        wdf["correct"] = wdf["pred"] == wdf["label"]

    with st.expander("Bộ lọc", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        fl_label = f1.multiselect("Nhãn thật", LABELS, LABELS, format_func=vi)
        fl_sym = f2.multiselect("Mã CK", sorted(df_all["symbol"].dropna().unique()))
        fl_src = f3.multiselect("Nguồn", sorted(df_all["source"].dropna().unique()))
        fl_diff = f4.selectbox("Difficult", ["Tất cả", "Khó (=1)", "Bình thường (=0)"])

        f5, f6, f7 = st.columns(3)
        if has_preds:
            fl_pred = f5.multiselect("Nhãn dự đoán", LABELS, LABELS, format_func=vi)
            fl_correct = f5.selectbox("Kết quả", ["Tất cả", "Đúng", "Sai"])
            conf_lo, conf_hi = f6.slider("Độ tự tin", 0.0, 1.0, (0.0, 1.0), 0.01)
        else:
            fl_pred, fl_correct = LABELS, "Tất cả"
            conf_lo, conf_hi = 0.0, 1.0
        search_q = f7.text_input("Tìm kiếm")

        sort_opts = ["published_at", "text_len"] + (["confidence", "entropy"] if has_preds else [])
        sort_sel = st.selectbox("Sắp xếp", sort_opts)
        sort_asc = st.checkbox("Tăng dần", value=False)

    mask = wdf["label"].isin(fl_label)
    if fl_sym:
        mask &= wdf["symbol"].isin(fl_sym)
    if fl_src:
        mask &= wdf["source"].isin(fl_src)
    if fl_diff == "Khó (=1)":
        mask &= wdf["difficult"] == 1
    elif fl_diff == "Bình thường (=0)":
        mask &= wdf["difficult"] == 0
    if has_preds:
        mask &= wdf["pred"].isin(fl_pred)
        if fl_correct == "Đúng":
            mask &= wdf["correct"]
        elif fl_correct == "Sai":
            mask &= ~wdf["correct"]
        mask &= (wdf["confidence"] >= conf_lo) & (wdf["confidence"] <= conf_hi)
    if search_q:
        mask &= (wdf["passage_text"].str.contains(search_q, case=False, na=False) |
                 wdf["title"].str.contains(search_q, case=False, na=False))

    fdf = wdf[mask].sort_values(sort_sel, ascending=sort_asc)
    st.write(f"**{len(fdf)}** mẫu /{len(wdf)}")

    # Confusion matrix + metrics (valid labels only)
    fdf_valid = fdf[fdf["valid_label"]] if "valid_label" in fdf.columns else fdf
    if has_preds and len(fdf_valid) >= 2:
        from sklearn.metrics import classification_report
        if len(fdf_valid) < len(fdf):
            st.caption(f"{len(fdf) - len(fdf_valid)} mẫu có nhãn skip/unclear — loại khỏi metrics.")
        col_cm, col_met = st.columns(2)

        with col_cm:
            fig_cm = confusion_fig(fdf_valid["label"].tolist(), fdf_valid["pred"].tolist())
            ev = st.plotly_chart(fig_cm, width="stretch",
                                 on_select="rerun", key="cm_ev")
            if ev and ev.selection and ev.selection.get("points"):
                pt = ev.selection["points"][0]
                inv = {v: k for k, v in LABEL_VI.items()}
                pred_c, true_c = inv.get(pt.get("x")), inv.get(pt.get("y"))
                if pred_c and true_c:
                    drill = fdf_valid[(fdf_valid["label"] == true_c) & (fdf_valid["pred"] == pred_c)]
                    st.info(f"Thật={vi(true_c)}, Dự đoán={vi(pred_c)} → **{len(drill)} mẫu**")
                    fdf = drill

        with col_met:
            rpt = classification_report(fdf_valid["label"], fdf_valid["pred"], labels=LABELS,
                                        zero_division=0, output_dict=True)
            rows = [{"Nhãn": vi(l),
                     "Precision": f"{rpt.get(l,{}).get('precision',0):.3f}",
                     "Recall": f"{rpt.get(l,{}).get('recall',0):.3f}",
                     "F1": f"{rpt.get(l,{}).get('f1-score',0):.3f}",
                     "Support": int(rpt.get(l,{}).get('support',0))} for l in LABELS]
            rows.append({"Nhãn": "Macro avg",
                         **{k: f"{rpt['macro avg'][m]:.3f}" for k, m in
                            [("Precision","precision"),("Recall","recall"),("F1","f1-score")]},
                         "Support": int(rpt["macro avg"]["support"])})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            p2n = int(((fdf_valid["label"] == "positive") & (fdf_valid["pred"] == "negative")).sum())
            n2p = int(((fdf_valid["label"] == "negative") & (fdf_valid["pred"] == "positive")).sum())
            st.metric("Lỗi nghiêm trọng Tăng↔Giảm", p2n + n2p)

    # Sample table
    st.subheader("Danh sách mẫu")
    show_cols = ["symbol", "published_at", "label", "difficult", "passage_text"]
    rename = {"symbol": "Mã CK", "published_at": "Ngày", "label": "Nhãn thật",
              "difficult": "Khó", "passage_text": "Văn bản"}
    if has_preds:
        show_cols += ["pred", "confidence", "correct"]
        rename |= {"pred": "Dự đoán", "confidence": "Tự tin", "correct": "Đúng"}
    st.dataframe(fdf[show_cols].rename(columns=rename), width="stretch", height=380)


# ════════════════════════════════════════════════════════════════════════════
# WHAT-IF
# ════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    st.header("What-if Analysis")
    st.caption("Sửa câu và xem xác suất đổi như thế nào. Kiểm tra model hiểu nghĩa hay chỉ bắt từ khoá.")

    wa, wb = st.columns(2)
    with wa:
        st.subheader("Gốc")
        text_a = st.text_area("Văn bản gốc", height=150, key="wif_a",
                               placeholder="Dán văn bản gốc…")
    with wb:
        st.subheader("Sửa đổi")
        pc = st.columns(4)
        if pc[0].button("Thêm phủ định"):
            st.session_state["wif_b"] = "không " + st.session_state.get("wif_a", "")
        if pc[1].button("Đổi ±"):
            src = st.session_state.get("wif_a", "")
            src = re.sub(r"(\d+\.?\d*)\s*%", lambda m: f"{-float(m.group(1)):.1f}%", src)
            src = re.sub(r"\+(\d)", r"-\1", src)
            st.session_state["wif_b"] = src
        if pc[2].button("Đổi mã CK"):
            st.session_state["wif_b"] = re.sub(r"\b([A-Z]{2,4})\b", "XYZ",
                                                st.session_state.get("wif_a", ""))
        if pc[3].button("Copy gốc"):
            st.session_state["wif_b"] = st.session_state.get("wif_a", "")
        text_b = st.text_area("Văn bản sửa đổi", height=120, key="wif_b")

    if st.button("So sánh", type="primary", disabled=not (bool(text_a) and bool(text_b))):
        with st.spinner("Đang chạy…"):
            m = load_model(model_key)
            probs_a, _, _ = predict_single(m, text_a)
            probs_b, _, _ = predict_single(m, text_b)
        pred_a = LABELS[int(np.argmax(probs_a))]
        pred_b = LABELS[int(np.argmax(probs_b))]

        c1, c2 = st.columns(2)
        c1.caption(f"**Gốc** → {vi(pred_a)}")
        c1.plotly_chart(prob_chart(probs_a), width="stretch", key="wif_probs_a")
        c2.caption(f"**Sửa đổi** → {vi(pred_b)}")
        c2.plotly_chart(prob_chart(probs_b), width="stretch", key="wif_probs_b")

        if pred_a != pred_b:
            st.success(f"Nhãn thay đổi: **{vi(pred_a)} → {vi(pred_b)}**")
        else:
            st.info("Nhãn không thay đổi.")

        st.dataframe(pd.DataFrame({
            "Nhãn": [vi(l) for l in LABELS],
            "Gốc": [f"{p:.3f}" for p in probs_a],
            "Sửa đổi": [f"{p:.3f}" for p in probs_b],
            "Δ": [f"{probs_b[i] - probs_a[i]:+.3f}" for i in range(3)],
        }), width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ════════════════════════════════════════════════════════════════════════════
with tab_calib:
    st.header("Calibration & Confidence")
    all_probs = get_preds()

    if all_probs is None:
        st.info("Bấm 'Nạp predictions' ở sidebar.")
    else:
        arr = np.array(all_probs)
        pred_lbls = [LABELS[int(np.argmax(p))] for p in all_probs]
        confs = arr.max(axis=1)
        ents = np.array([calc_entropy(p) for p in all_probs])
        # only evaluate on rows with valid (3-class) labels
        valid_mask = df_all["valid_label"].values
        true_lbls_all = df_all["label"].tolist()
        ok = np.array([p == t for p, t in zip(pred_lbls, true_lbls_all)])
        ok[~valid_mask] = False  # skip/unclear rows never count as correct
        true_lbls = [l for l, v in zip(true_lbls_all, valid_mask) if v]
        all_probs_valid = [p for p, v in zip(all_probs, valid_mask) if v]
        pred_lbls_valid = [p for p, v in zip(pred_lbls, valid_mask) if v]
        ok_valid = ok[valid_mask]
        confs_valid = confs[valid_mask]
        ents_valid = ents[valid_mask]

        n_skipped = int((~valid_mask).sum())
        if n_skipped:
            st.caption(f"Metrics tính trên {valid_mask.sum()} mẫu hợp lệ ({n_skipped} skip/unclear loại ra).")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{ok_valid.mean():.1%}")
        m2.metric("Tự tin trung bình", f"{confs_valid.mean():.3f}")
        m3.metric("Entropy trung bình", f"{ents_valid.mean():.3f}")
        m4.metric("Sai nhưng tự tin > 90%", int(((confs_valid > 0.9) & ~ok_valid).sum()))
        st.divider()

        col_rel, col_hist = st.columns(2)
        with col_rel:
            st.subheader("Reliability Diagram")
            st.plotly_chart(reliability_fig(true_lbls, all_probs_valid), width="stretch",
                            key="calib_reliability")
        with col_hist:
            st.subheader("Phân bố độ tự tin")
            fig_h = go.Figure()
            for flag, color, name in [(True, "#0ca30c", "Đúng"), (False, "#d03b3b", "Sai")]:
                fig_h.add_trace(go.Histogram(x=confs_valid[ok_valid == flag], name=name,
                                             marker_color=color, opacity=0.65, nbinsx=20))
            fig_h.update_layout(barmode="overlay", xaxis_title="Độ tự tin", yaxis_title="Số mẫu",
                                 height=300, margin=dict(t=16, b=40, l=40, r=16),
                                 legend=dict(orientation="h"),
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_h, width="stretch", key="calib_conf_hist")

        st.divider()
        st.subheader("Abstention Threshold")
        thresh = st.slider("Abstain khi entropy >", 0.0, 1.0, 0.5, 0.01)
        keep = ents_valid <= thresh
        a1, a2, a3 = st.columns(3)
        a1.metric("Coverage", f"{keep.mean():.1%}")
        a2.metric("Accuracy (kept)", f"{ok_valid[keep].mean():.1%}" if keep.any() else "—")
        a3.metric("Abstained", int((~keep).sum()))

        from nlp_module.evaluate import abstention_curve
        curve = abstention_curve(true_lbls, all_probs_valid)
        cdf = pd.DataFrame(curve, columns=["h_thresh", "coverage", "accuracy"])
        fig_c = go.Figure(go.Scatter(x=cdf["coverage"], y=cdf["accuracy"],
                                     mode="lines+markers", line=dict(color="#6366f1")))
        fig_c.add_vline(x=float(keep.mean()), line_dash="dash", line_color="rgba(128,128,128,.5)")
        fig_c.update_layout(xaxis_title="Coverage", yaxis_title="Accuracy",
                             height=270, margin=dict(t=16, b=40, l=55, r=16),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, width="stretch", key="calib_coverage")

        st.divider()
        st.subheader("Sai nhưng rất tự tin (confidence > 0.8)")
        # use full df_all but only valid rows
        valid_df = df_all[valid_mask].copy()
        valid_df["pred"] = np.array(pred_lbls_valid)
        valid_df["confidence"] = confs_valid
        hce = valid_df[(confs_valid > 0.8) & ~ok_valid]
        if len(hce):
            st.dataframe(
                hce[["symbol", "published_at", "label", "pred", "confidence", "passage_text"]]
                .sort_values("confidence", ascending=False),
                width="stretch", height=300)
        else:
            st.success("Không có mẫu nào sai nhưng tự tin > 0.8")


# ════════════════════════════════════════════════════════════════════════════
# PER-TICKER DETAIL
# ════════════════════════════════════════════════════════════════════════════
with tab_stock:
    st.header("Chi tiết theo mã cổ phiếu")

    vers = db_versions()
    if not vers:
        st.error("Chưa có kết quả nào trong `data/nlp.db`. Chạy `nlp_module process --model ...` trước.")
        st.stop()

    hc = st.columns([2, 2, 3])
    ver = hc[0].selectbox("Model version (trong DB)", vers,
                          index=vers.index("baseline-tfidf-lr-v1") if "baseline-tfidf-lr-v1" in vers else 0,
                          help="Phiên bản model đã cho điểm các bài báo. Khác với model ở sidebar, "
                               "cái đó chỉ dùng cho inference trực tiếp ở các tab khác.")
    res = db_results(ver)
    syms = sorted(res["symbol"].unique())
    sym = hc[1].selectbox("Mã cổ phiếu", syms, index=syms.index("VIC") if "VIC" in syms else 0)
    only_direct = hc[2].checkbox(
        "Chỉ tin nói trực tiếp về công ty (relation = direct)", value=True,
        help="Bài liệt kê 5+ công ty (bảng lãi suất, VN30, roundup BCTC) được gán relation=market và "
             "bị loại khỏi mọi feature. Bỏ tick để xem cả chúng.")

    d = res[res["symbol"] == sym]
    d_feat = d[d["relation"] != "market"]          # exactly what build_snapshot() admits
    d_show = d[d["relation"] == "direct"] if only_direct else d

    if d.empty:
        st.warning(f"Không có bài nào cho {sym} với model {ver}.")
        st.stop()

    # ── what the numbers are, and what they are not ─────────────────────────
    with st.expander("Đọc kỹ: những con số dưới đây nghĩa là gì — và KHÔNG phải gì", expanded=False):
        st.markdown("""
**Model trả lời đúng một câu hỏi:** *đoạn văn này nói gì về triển vọng / kết quả của **chính công ty đó**?*

Nó **không** dự báo giá. Theo guideline, các loại tin sau là **Trung tính** dù trông tích cực hay tiêu cực:

- giá chạy, khối lượng, khối ngoại mua/bán ròng
- khuyến mãi, ra mắt sản phẩm, tài trợ, bảng lãi suất tiền gửi
- phát biểu lạc quan của lãnh đạo (khi chưa kèm số liệu)

Mã chỉ xuất hiện như **khoản đầu tư / đối tác / mốc địa lý** của công ty khác thì đúng ra phải bị loại (`skip`).

**Chất lượng đo được trên 240 nhãn người (stream `eval`, đã đóng băng):**

| | |
|---|---|
| macro-F1 | **0.626** `[CI 0.46 – 0.74]` |
| F1 từng lớp | Tích cực 0.53 · Trung tính 0.85 · Tiêu cực 0.50 |
| AUC phân biệt tích cực vs tiêu cực | **0.978** |

Đọc: **phân biệt tốt/xấu gần như hoàn hảo**; điểm yếu là biết *khi nào* một đoạn có chiều, và nó
**gọi thiếu tích cực** (recall 0.48). F1 lớp Tiêu cực dựa trên **4 mẫu** — đừng tin con số đó.
""")

    # ── headline state ──────────────────────────────────────────────────────
    st.subheader(f"{sym} — trạng thái hiện tại")
    last = d_feat["published_at"].max() if not d_feat.empty else None
    w72 = d_feat[d_feat["published_at"] >= last - pd.Timedelta(hours=72)] if last is not None else d_feat.iloc[:0]
    kept = w72[~w72["uncertain"]]                  # uncertain is never counted as neutral

    m = st.columns(5)
    m[0].metric("Bài (toàn bộ lịch sử)", len(d_feat),
                help="Đã lọc trùng lặp (chỉ bản đại diện của mỗi cụm tin lặp lại).")
    m[1].metric("Tin gần nhất", last.strftime("%d/%m/%Y") if last is not None else "—")
    m[2].metric("Sentiment 72h", f"{kept['score'].mean():+.3f}" if len(kept) else "—",
                help="Trung bình (p_tích_cực − p_tiêu_cực) trong 72h kể từ bài gần nhất, "
                     "đã bỏ các dòng `uncertain`. Thang [−1, +1]. '—' nghĩa là KHÔNG CÓ DỮ LIỆU, "
                     "không phải trung tính.")
    m[3].metric("Bỏ qua vì thiếu tự tin", f"{w72['uncertain'].mean():.0%}" if len(w72) else "—",
                help="Tỷ lệ cửa sổ 72h bị abstain (entropy vượt ngưỡng đã chọn trên validation).")
    m[4].metric("Bằng chứng yếu", f"{w72['difficult'].mean():.0%}" if len(w72) else "—",
                help="Passage có công ty khác lẫn vào, hoặc chỉ có title mà không có câu nào trong thân bài.")

    if len(kept) == 0 and len(w72) > 0:
        st.warning("Cả cửa sổ 72h đều bị abstain → **không có** giá trị sentiment. "
                   "Đây là hành vi thiết kế: thiếu tự tin không bị quy thành trung tính.")

    # ── last known market price, from storage only (no network on a rerun) ──
    qh_all = db_quotes()
    qs_ = qh_all[qh_all["symbol"] == sym] if not qh_all.empty else qh_all
    if not qs_.empty:
        lq = qs_.iloc[-1]
        p_ = st.columns(4)
        p_[0].metric("Giá khớp (SSI)", f"{lq['matched_price']:,.0f}",
                     f"{lq['price_change_pct']:+.2f}%")
        p_[1].metric("Tham chiếu", f"{lq['ref_price']:,.0f}")
        p_[2].metric("Phiên", str(lq["session"] or "—"))
        p_[3].metric("Đọc lúc", f"{lq['fetched_at']:%d/%m %H:%M} UTC")
        st.caption("Giá là bản đọc đã lưu gần nhất, không phải realtime. Cập nhật ở tab "
                   "**Sentiment vs Giá SSI** hoặc bằng `nlp_module quotes`. "
                   "Sentiment ở trên **không** dự báo con số này.")
    else:
        st.caption("Chưa có giá đã lưu cho mã này — xem tab **Sentiment vs Giá SSI**.")

    # ── history ─────────────────────────────────────────────────────────────
    st.subheader("Diễn biến theo thời gian")
    snaps = db_snapshots(ver)
    if snaps is not None:
        sd = snaps[(snaps["symbol"] == sym) & snaps["has_news"]].sort_values("forecast_cutoff")
        px_ = load_prices()
        px_ = px_[px_["symbol"] == sym]
        # three panels on one time axis instead of a dual-axis plot: each measure keeps its own honest scale
        rows_ = 3 if not px_.empty else 2
        fig = make_subplots(rows=rows_, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.5, 0.2, 0.3][:rows_],
                            subplot_titles=["Sentiment 72h  (trên 0 = tin tích cực, dưới 0 = tiêu cực)",
                                            "Số tin trong 72h", "Giá đóng cửa (nghìn VND)"][:rows_])
        x_, s_ = sd["forecast_cutoff"], sd["sentiment_mean"]
        for part, lbl in ((s_.clip(lower=0), "positive"), (s_.clip(upper=0), "negative")):
            fig.add_scatter(x=x_, y=part, mode="lines", fill="tozeroy", connectgaps=False,
                            line=dict(color=LABEL_COLOR[lbl], width=1), name=vi(lbl), row=1, col=1,
                            customdata=np.c_[s_, sd["news_count_72h"]],
                            hovertemplate="%{x|%d/%m/%Y}<br>sentiment %{customdata[0]:+.3f}"
                                          "<br>%{customdata[1]} tin<extra></extra>")
        fig.add_hline(y=0, line_color="rgba(128,128,128,.6)", line_width=1, row=1, col=1)
        fig.add_bar(x=x_, y=sd["news_count_72h"], marker_color=LABEL_COLOR["neutral"], name="Số tin",
                    showlegend=False, hovertemplate="%{x|%d/%m/%Y}: %{y} tin<extra></extra>", row=2, col=1)
        if rows_ == 3:
            fig.add_scatter(x=px_["date"], y=px_["close"], mode="lines", name="Giá", showlegend=False,
                            line=dict(color="#52514e", width=1.4),
                            hovertemplate="%{x|%d/%m/%Y}: %{y:,.2f}<extra></extra>", row=3, col=1)
        lim_ = max(0.3, float(s_.abs().max()) * 1.1) if s_.notna().any() else 1
        fig.update_yaxes(range=[-lim_, lim_], row=1, col=1)   # symmetric: 0 stays in the middle
        fig.update_xaxes(rangeselector=dict(buttons=[dict(count=3, label="3T", step="month", stepmode="backward"),
                                                     dict(count=6, label="6T", step="month", stepmode="backward"),
                                                     dict(count=1, label="1N", step="year", stepmode="backward"),
                                                     dict(label="Tất cả", step="all")], y=1.08),
                         row=1, col=1)
        x_end = x_.max() if len(x_) else pd.Timestamp.now(tz="UTC")
        fig.update_xaxes(range=[x_end - pd.DateOffset(years=1), x_end])   # last year by default; 'Tất cả' widens
        fig.update_layout(height=560, margin=dict(t=60, b=30, l=55, r=20), hovermode="x unified",
                          legend=dict(orientation="h", x=1, xanchor="right", y=1.1), bargap=0.1)
        st.plotly_chart(fig, width="stretch", key="stock_series")
        st.caption("Vùng trống = ngày không có tin trong 72h (hoặc cả cửa sổ bị abstain): **thiếu dữ liệu**, "
                   "không phải sentiment 0. Ba ô dùng chung trục thời gian — rê chuột để xem cùng một ngày. "
                   + ("" if rows_ == 3 else f"`data/prices.csv` không có giá cho {sym}."))
    else:
        st.info(f"Chưa có file snapshot cho `{ver}`. Chạy "
                f"`nlp_module snapshot --model ... --mode backtest ...` để xem chuỗi thời gian.")

    mix = (d_show.assign(month=d_show["published_at"].dt.tz_convert(None).dt.to_period("M").astype(str))
           .groupby(["month", "pred"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0))
    if not mix.empty:
        figm = go.Figure()
        for l in LABELS:
            figm.add_bar(x=mix.index, y=mix[l], name=vi(l), marker_color=LABEL_COLOR[l])
        figm.update_layout(barmode="stack", height=260, margin=dict(t=16, b=40, l=45, r=16),
                           yaxis_title="Số bài", legend=dict(orientation="h", y=1.18),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figm, width="stretch", key="stock_mix")

    # ── article-level evidence ──────────────────────────────────────────────
    st.subheader(f"Từng bài báo ({len(d_show)} bài)")
    fc = st.columns(3)
    lbl_f = fc[0].multiselect("Nhãn", [vi(l) for l in LABELS], default=[vi(l) for l in LABELS])
    flag_f = fc[1].multiselect("Chỉ hiện", ["Cần review", "Thiếu tự tin", "Bằng chứng yếu"], default=[])
    n_show = fc[2].slider("Số bài hiển thị", 5, 100, 20)

    inv = {v: k for k, v in LABEL_VI.items()}
    t = d_show[d_show["pred"].isin([inv[x] for x in lbl_f])] if lbl_f else d_show.iloc[:0]
    if "Cần review" in flag_f:
        t = t[t["needs_review"]]
    if "Thiếu tự tin" in flag_f:
        t = t[t["uncertain"]]
    if "Bằng chứng yếu" in flag_f:
        t = t[t["difficult"]]

    st.caption(f"{len(t)} bài khớp bộ lọc. Bảng đã sắp theo thời gian giảm dần.")
    for r in t.head(n_show).itertuples():
        flags = " ".join(f"`{x}`" for x, on in
                         [("thiếu tự tin", r.uncertain), ("bằng chứng yếu", r.difficult),
                          ("cần review", r.needs_review), (r.relation, r.relation != "direct")] if on)
        head = (f"{r.published_at:%d/%m/%Y %H:%M} · **{vi(r.pred)}** "
                f"({max(r.p_pos, r.p_neu, r.p_neg):.0%}) · {r.title[:85]}")
        with st.expander(head):
            pc = st.columns([2, 3])
            pc[0].plotly_chart(prob_chart((r.p_pos, r.p_neu, r.p_neg)), width="stretch",
                               key=f"stock_prob_{r.Index}")
            pc[1].markdown(f"""
**Đoạn văn model thực sự đọc** ({len(r.passage_text)} ký tự):

> {r.passage_text}

- entropy **{r.entropy:.3f}** · relation **{r.relation}** · matching_score **{r.matching_score:.2f}**
- nguồn: {r.source} · [mở bài gốc]({r.url})
{("- cờ: " + flags) if flags else ""}
""")
            if r.difficult and len(r.passage_text) < 140:
                pc[1].warning("Passage gần như chỉ có title — model không tìm được câu nào trong thân bài "
                              "nhắc tới mã này. Cùng một bài báo có thể cho mã khác điểm khác hẳn.")

    # ── the forecast question, answered honestly ────────────────────────────
    st.subheader("Dự báo giá?")
    st.error("""**Hệ thống này không xuất ra dự báo giá, và cố tình không làm vậy.**

Đã test 4 cách định nghĩa mục tiêu trên 10 mã có dữ liệu giá. Thêm feature text vào feature giá
không cải thiện gì ở bất kỳ cách nào:""")
    st.dataframe(pd.DataFrame([
        {"Mục tiêu dự báo": "Chiều giá T+2", "AUC chỉ giá": 0.5189,
         "AUC giá + text": 0.5140, "Chênh [95% CI]": "−0.005 [−0.021, +0.012]"},
        {"Mục tiêu dự báo": "Chiều giá T+5", "AUC chỉ giá": 0.5463,
         "AUC giá + text": 0.5326, "Chênh [95% CI]": "−0.014 [−0.030, +0.005]"},
        {"Mục tiêu dự báo": "Vượt thị trường (excess) T+5", "AUC chỉ giá": 0.5330,
         "AUC giá + text": 0.5236, "Chênh [95% CI]": "−0.010 [−0.026, +0.007]"},
        {"Mục tiêu dự báo": "Độ lớn biến động T+5", "AUC chỉ giá": 0.5814,
         "AUC giá + text": 0.5815, "Chênh [95% CI]": "+0.000 [−0.012, +0.012]"},
    ]), hide_index=True, width="stretch")
    st.caption("AUC 0.50 = đoán bừa. `news_count_72h` một mình có AUC 0.558 với độ lớn biến động — "
               "tin tức CÓ dự báo được mức biến động — nhưng `vol10` (độ lệch chuẩn 10 ngày) đạt 0.582 "
               "và đã bao trùm hết: thêm cả 8 feature text vào vol10 cho 0.582 → 0.581. "
               "Giá trị của module này là **giải thích vì sao một mã đang lên báo**, không phải alpha.")


# ════════════════════════════════════════════════════════════════════════════
# SENTIMENT vs SSI PRICE BOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_price:
    st.header("Sentiment của model vs bảng giá điện tử SSI")
    st.caption("Nguồn giá: bảng giá iBoard của SSI (`iboard-query.ssi.com.vn`), đọc 1 request cho cả sàn, "
               "tôn trọng robots.txt. Giá tính bằng **VND**.")

    vers_p = db_versions()
    if not vers_p:
        st.error("Chưa có kết quả model nào trong `data/nlp.db`.")
        st.stop()

    tc = st.columns([2, 3, 2])
    ver_p = tc[0].selectbox("Model version", vers_p,
                            index=vers_p.index("baseline-tfidf-lr-v1") if "baseline-tfidf-lr-v1" in vers_p else 0,
                            key="price_ver")
    all_syms = sorted(db_results(ver_p)["symbol"].unique())
    default_syms = [s_ for s_ in ["ACB", "BID", "CTG", "FPT", "HDB", "HPG", "MSN", "MWG",
                                  "NVL", "STB", "TCB", "VCB", "VHM", "VIC", "VPB"] if s_ in all_syms]
    syms_p = tc[1].multiselect("Mã cần so sánh", all_syms, default=default_syms, key="price_syms")
    if tc[2].button("Lấy giá SSI ngay", type="primary", width="stretch"):
        if not syms_p:
            st.warning("Chọn ít nhất một mã.")
        else:
            with st.spinner("Đang đọc bảng giá SSI…"):
                try:
                    st.session_state["cmp_df"] = fetch_quotes_now(syms_p, ver_p)
                    st.session_state["cmp_ver"] = ver_p
                    db_quotes.clear()
                except Exception as e:
                    st.error(f"Không đọc được bảng giá SSI: `{type(e).__name__}: {e}`")

    st.warning("""**Đây là bảng theo dõi, KHÔNG phải thước đo độ chính xác.**
Model không dự báo giá — nó đánh giá tin tức nói gì về triển vọng công ty. Mọi ablation đã đo đều cho
AUC trong vùng nhiễu quanh 0.5 (xem tab Theo mã CK → mục *Dự báo giá?*). Cột "cùng dấu" là **đồng biến quan sát được**,
không phải tỷ lệ đúng. Với 7 mã so được, chênh lệch vài phần trăm là nhiễu thuần.""")

    cmp_df = st.session_state.get("cmp_df")
    if cmp_df is None or st.session_state.get("cmp_ver") != ver_p:
        cmp_df, read_at = latest_cmp(ver_p, tuple(syms_p))
        if read_at is not None and not cmp_df.empty:
            st.caption(f"Bản đọc giá đã lưu lúc **{read_at.tz_convert('Asia/Ho_Chi_Minh'):%d/%m %H:%M}** "
                       "(cron `scripts/live_update.sh`, 15 phút/lần trong giờ giao dịch). "
                       "Sentiment tính tới hiện tại.")
    if cmp_df.empty:
        st.info("Chưa có bản đọc giá nào cho các mã đã chọn — bấm **Lấy giá SSI ngay**.")
    else:
        mkt = float(cmp_df["market_change_pct"].iloc[0])
        sess = cmp_df["session"].mode()
        both_raw = cmp_df[cmp_df["same_sign"].notna()]
        both_ex = cmp_df[cmp_df["same_sign_excess"].notna()]

        k = st.columns(5)
        k[0].metric("Thị trường (bình quân đều)", f"{mkt:+.2f}%",
                    help="Trung bình % thay đổi của các mã đã chọn có khớp lệnh. Dùng làm proxy thị trường, "
                         "giống `trial_forecast --target excess`.")
        k[1].metric("Phiên", sess.iloc[0] if len(sess) else "—",
                    help="Mã phiên của SSI: P=trước giờ mở, O/K=liên tục, A=ATC, C=đã đóng.")
        k[2].metric("Mã có cả sentiment & giá", f"{len(both_raw)}/{len(cmp_df)}",
                    help="Mã không có tin trong 72h, hoặc cả cửa sổ bị abstain, thì KHÔNG có sentiment — "
                         "và không so được. Đó là thiếu dữ liệu, không phải trung tính.")
        k[3].metric("Cùng dấu — giá thô",
                    f"{int(both_raw['same_sign'].sum())}/{len(both_raw)}" if len(both_raw) else "—")
        k[4].metric("Cùng dấu — so với thị trường",
                    f"{int(both_ex['same_sign_excess'].sum())}/{len(both_ex)}" if len(both_ex) else "—",
                    help="Có ý nghĩa hơn cột bên cạnh: khi cả sàn cùng chạy một hướng, dấu của giá thô "
                         "chủ yếu phản ánh nhân tố thị trường, không phải tin riêng của mã.")

        if len(both_raw) and len(both_ex) and both_raw["same_sign"].mean() < both_ex["same_sign_excess"].mean():
            st.info(f"Hôm nay cả thị trường đi {'xuống' if mkt < 0 else 'lên'} ({mkt:+.2f}%), nên dấu của giá thô "
                    f"gần như chỉ nói về thị trường. Cột *so với thị trường* mới là so sánh đúng.")

        fig = go.Figure()
        sc = cmp_df.dropna(subset=["sentiment_mean", "excess_change_pct"])
        if not sc.empty:
            fig.add_scatter(x=sc["sentiment_mean"], y=sc["excess_change_pct"], mode="markers+text",
                            text=sc["symbol"], textposition="top center",
                            customdata=np.c_[sc["news_count_72h"], sc["price_change_pct"]],
                            marker=dict(size=10 + 4 * np.sqrt(sc["news_count_72h"]), color=sc["sentiment_mean"],
                                        colorscale=DIVERGING, cmin=-0.5, cmax=0.5, opacity=.9,
                                        line=dict(width=2, color="rgba(255,255,255,.9)"),
                                        colorbar=dict(title="Sentiment", thickness=10, len=.6)),
                            name="Mã CK", hovertemplate="<b>%{text}</b><br>sentiment %{x:+.3f} · %{customdata[0]} tin"
                                                        "<br>giá %{customdata[1]:+.2f}% · so thị trường %{y:+.2f}%"
                                                        "<extra></extra>")
            # quadrant names, so the reader does not have to work out what each corner means
            for qx, qy, txt in ((1, 1, "Tin tốt · giá mạnh hơn TT"), (0, 1, "Tin xấu · giá vẫn mạnh hơn TT"),
                                (0, 0, "Tin xấu · giá yếu hơn TT"), (1, 0, "Tin tốt · giá vẫn yếu hơn TT")):
                fig.add_annotation(xref="x domain", yref="y domain", x=qx, y=qy, text=txt, showarrow=False,
                                   xanchor="right" if qx else "left", yanchor="top" if qy else "bottom",
                                   font=dict(size=11, color="#898781"))
            lim = max(0.1, float(sc["sentiment_mean"].abs().max()) * 1.25)
            fig.update_xaxes(range=[-lim, lim])
        fig.add_hline(y=0, line_color="rgba(128,128,128,.6)", line_width=1)
        fig.add_vline(x=0, line_color="rgba(128,128,128,.6)", line_width=1)
        fig.update_layout(height=440, margin=dict(t=20, b=45, l=60, r=20), showlegend=False,
                          xaxis_title="Sentiment 72h của model  (← tiêu cực · tích cực →)",
                          yaxis_title="Giá so với thị trường (%)")
        st.plotly_chart(fig, width="stretch", key="price_scatter")
        st.caption("Kích thước bóng = số tin trong 72h (bóng to = sentiment dựa trên nhiều tin hơn). "
                   "Nếu sentiment có thông tin, điểm sẽ dồn vào góc trên-phải và dưới-trái. "
                   "Hiện tại chúng phân tán — đúng như AUC đã đo.")

        show = cmp_df.assign(
            Mã=cmp_df["symbol"],
            Sentiment=cmp_df["sentiment_mean"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "— (không có tin)"),
            Tin72h=cmp_df["news_count_72h"],
            Abstain=cmp_df["uncertain_ratio"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—"),
            Yếu=cmp_df["difficult_ratio"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "—"),
            Giá=cmp_df["matched_price"].map(lambda v: f"{v:,.0f}"),
            Thay_đổi=cmp_df["price_change_pct"].map(lambda v: f"{v:+.2f}%"),
            So_thị_trường=cmp_df["excess_change_pct"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"),
            Cùng_dấu=cmp_df["same_sign_excess"].map({True: "Có", False: "Không"}).fillna("—"),
        )[["Mã", "Sentiment", "Tin72h", "Abstain", "Yếu", "Giá", "Thay_đổi", "So_thị_trường", "Cùng_dấu"]]
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("`Abstain` = phần cửa sổ 72h model tự nhận không đủ tự tin. `Yếu` = passage có công ty khác "
                   "lẫn vào hoặc chỉ có title. Hai cột này nói độ tin của Sentiment bên cạnh.")

    # ── tracking over time ──────────────────────────────────────────────────
    st.subheader("Lịch sử theo dõi")
    qh = db_quotes()
    if qh.empty:
        st.info("Chưa có bản ghi nào. Mỗi lần bấm **Lấy giá SSI ngay** (hoặc chạy "
                "`nlp_module -v quotes`) sẽ lưu một lần đọc vào bảng `market_quote`.")
    else:
        st.caption(f"{len(qh)} bản ghi · {qh['symbol'].nunique()} mã · "
                   f"{qh['fetched_at'].min():%d/%m/%Y %H:%M} → {qh['fetched_at'].max():%d/%m/%Y %H:%M} UTC")
        hsym = st.selectbox("Mã", sorted(qh["symbol"].unique()), key="qhist_sym")
        h = qh[qh["symbol"] == hsym]
        if len(h) < 2:
            st.info(f"Chỉ có {len(h)} bản ghi cho {hsym}. Cần ít nhất 2 lần đọc để vẽ diễn biến — "
                    "chạy `nlp_module quotes` định kỳ (ví dụ qua `/loop` hoặc cron).")
        figh = go.Figure()
        figh.add_scatter(x=h["fetched_at"], y=h["matched_price"], mode="lines+markers",
                         name="Giá khớp", line=dict(color="#2563eb", width=1.6))
        figh.add_scatter(x=h["fetched_at"], y=h["ref_price"], mode="lines",
                         name="Giá tham chiếu", line=dict(color="rgba(148,163,184,.9)", dash="dot"))
        figh.update_layout(height=280, margin=dict(t=16, b=40, l=60, r=20), yaxis_title="VND",
                           legend=dict(orientation="h", y=1.2),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figh, width="stretch", key="qhist_chart")
        st.caption("Để theo dõi liên tục: `nlp_module -v quotes --symbols VIC VHM …` ghi thêm một bản ghi "
                   "mỗi lần chạy, kèm log một dòng cho mỗi mã (sentiment vs giá thật vs so với thị trường).")


# ════════════════════════════════════════════════════════════════════════════
# VOLATILITY FORECAST (reads vol_forecast + walk-forward report; writing is `python -m nlp_module.volcast`)
# ════════════════════════════════════════════════════════════════════════════
with tab_vol:
    st.header("Dự báo biến động")
    st.caption("Dự báo **biên độ** (biến động mạnh hơn trung vị 15 mã), **không phải chiều tăng/giảm**. "
               "Model chỉ đưa ra nhận định khi vượt ngưỡng tự tin chọn trên validation; "
               "độ chính xác thật đã đo xem bảng bên dưới. Còn lại hiển thị \"—\".")
    _vf = db_vol_forecast(volcast.VERSION)
    _rep = vol_report()
    if _vf.empty or not _rep or not (ROOT / volcast.BUNDLE).exists():
        st.info("Chưa có dự báo biến động (thiếu model, báo cáo hoặc bảng `vol_forecast` còn rỗng). Chạy:\n\n"
                "```\npython -m nlp_module.volcast train\npython -m nlp_module.volcast predict\n```")
    else:
        _lbl = {"strong": "Mạnh", "weak": "Yếu"}
        _vf["cell"] = [f"{_lbl.get(c, '—')} ({p:.2f})" for c, p in zip(_vf["call"], _vf["p_strong"])]
        _piv = _vf.pivot(index="symbol", columns="h", values="cell").reindex(columns=[2, 5]).fillna("—")
        _piv.columns = [f"T+{h}" for h in _piv.columns]
        _bd = _vf.groupby("symbol")["base_date"].max().rename("Phiên căn cứ")
        _piv = _piv.join(_bd).rename_axis("Mã")
        n_call = int(_vf["call"].notna().sum())
        _vc = lambda v: f"{v:.1%}".replace(".", ",")   # Vietnamese decimal comma
        for _row in (r for r in _rep["rows"] if r["fold"] == "C"):   # live fold: warn above the table, never suppress calls
            if _row["threshold"] is None or (_row["test_acc"] or 0) < 0.65:
                _why = ("không có ngưỡng đủ tin cậy" if _row["threshold"] is None else
                        f"{_vc(_row['test_acc'])} ở độ phủ {_vc(_row['test_cov'])}, n={_row['test_n']}")
                st.error(f"Fold đang chạy thật (C), T+{_row['h']}: {_why} — chưa đạt mục tiêu 0,65; "
                         "xem nhận định dưới đây như tham khảo.")
        st.subheader(f"Dự báo mới nhất ({n_call}/{len(_vf)} ô có nhận định)")
        st.dataframe(_piv, width="stretch")
        st.caption("Ô: nhận định (xác suất biến động mạnh). Xác suất không đủ xa ngưỡng → \"—\".")

        st.subheader("Độ tin cậy đã đo")
        st.caption("Walk-forward trên dữ liệu chưa thấy. **Fold C là fold chạy thật**: model và ngưỡng của fold C "
                   "được dùng cho dự báo ở trên. Mọi độ chính xác phải đọc cùng độ phủ (tỉ lệ ngày có nhận định) "
                   "và n (số nhận định). \"Quy tắc đơn giản\" = \"biến động 10 phiên cao → Mạnh\", cùng số nhận định.")
        _r = pd.DataFrame(_rep["rows"])
        _r["bad"] = _r["threshold"].isna() | (_r["test_acc"].fillna(0) < 0.65)
        _show = pd.DataFrame({
            "Fold": _r["fold"], "Kỳ hạn": "T+" + _r["h"].astype(str), "Model": _r["model"].fillna("—"),
            "Độ chính xác (test)": _r["test_acc"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
            "Độ phủ": _r["test_cov"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
            "n": _r["test_n"].map(lambda v: "—" if pd.isna(v) else f"{int(v)}"),
            "CI 95%": [("—" if pd.isna(a) else f"{a:.1%} – {b:.1%}") for a, b in zip(_r["ci_lo"], _r["ci_hi"])],
            "Quy tắc đơn giản": _r["rule_acc"].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
            "CI 95% quy tắc": [("—" if pd.isna(a) else f"{a:.1%} – {b:.1%}") for a, b in zip(_r["rule_ci_lo"], _r["rule_ci_hi"])],
        })
        st.dataframe(_show.style.apply(
            lambda x: ["background-color: rgba(227,73,72,.22)" if _r["bad"].iloc[i] else "" for i in range(len(x))],
            axis=0), width="stretch", hide_index=True)
        for _, row in _r[_r["bad"]].iterrows():
            why = ("không có ngưỡng đủ tin cậy (không đưa ra nhận định)" if pd.isna(row["threshold"])
                   else f"độ chính xác test {row['test_acc']:.1%} < 65% "
                        f"(độ phủ {row['test_cov']:.1%}, n={int(row['test_n'])})")
            st.warning(f"Fold {row['fold']} · T+{int(row['h'])}: {why}.")
        st.caption(f"Phiên bản `{_rep.get('version')}` · tạo lúc {_rep.get('created_at')}. "
                   "Đây là công cụ tham khảo biên độ, không phải khuyến nghị đầu tư.")


# ════════════════════════════════════════════════════════════════════════════
# 3D OVERVIEW: ticker x time x sentiment
# ════════════════════════════════════════════════════════════════════════════
with tab_3d:
    st.header("Toàn cảnh sentiment: mã × thời gian")
    st.caption("Mỗi điểm là một mã trong một kỳ: **chiều cao** = sentiment trung bình của các tin trong kỳ, "
               "**màu** = cùng giá trị đó (xanh tích cực, đỏ tiêu cực, xám trung tính), **kích thước bóng** = số tin. "
               "Mỗi mã là một dải chạy theo thời gian. "
               "Mặt phẳng mờ là mức 0. Kéo để xoay, cuộn để phóng to, rê chuột để xem số liệu.")
    vers_3 = db_versions()
    oc = st.columns([2, 2, 2, 4])
    ver_3 = oc[0].selectbox("Model version", vers_3, key="v3_ver",
                            index=vers_3.index("baseline-tfidf-lr-v1") if "baseline-tfidf-lr-v1" in vers_3 else 0)
    gran = oc[1].radio("Gom theo", ["Tuần", "Tháng"], index=1, horizontal=True, key="v3_gran")
    span = oc[2].select_slider("Khoảng thời gian", ["3 tháng", "6 tháng", "1 năm", "2 năm", "Tất cả"],
                               value="1 năm", key="v3_span")
    r3 = db_results(ver_3)
    r3 = r3[(r3["relation"] != "market") & ~r3["uncertain"]]   # the rows that feed sentiment_mean
    if span != "Tất cả":
        months = {"3 tháng": 3, "6 tháng": 6, "1 năm": 12, "2 năm": 24}[span]
        r3 = r3[r3["published_at"] >= r3["published_at"].max() - pd.DateOffset(months=months)]
    by_n = r3["symbol"].value_counts()
    syms3 = oc[3].multiselect("Mã", sorted(by_n.index), default=sorted(by_n.index[:12]), key="v3_syms",
                              help="Mặc định: 12 mã có nhiều tin nhất trong khoảng đã chọn.")
    r3 = r3[r3["symbol"].isin(syms3)]
    if r3.empty:
        st.info("Không có tin nào khớp bộ lọc.")
    else:
        g = (r3.assign(t=r3["published_at"].dt.tz_convert(None).dt.to_period("W" if gran == "Tuần" else "M")
                       .dt.start_time,
                       pos=r3["pred"] == "positive", neg=r3["pred"] == "negative")
             .groupby(["symbol", "t"]).agg(score=("score", "mean"), n=("score", "size"),
                                           pos=("pos", "mean"), neg=("neg", "mean")).reset_index())
        # rows sorted by overall sentiment, so the most positive tickers sit together at the back
        order = g.groupby("symbol")["score"].mean().sort_values().index.tolist()
        g["yi"] = g["symbol"].map({s_: i for i, s_ in enumerate(order)})
        t0 = g["t"].min()
        g["xd"] = (g["t"] - t0).dt.days     # numeric x: the zero plane and stems need plain numbers
        ticks = pd.date_range(t0, g["t"].max(), freq="MS")
        ticks = ticks[::max(1, len(ticks) // 8)]
        tickvals, ticktext = (ticks - t0).days, [f"{t_:%m/%Y}" for t_ in ticks]
        fmt = "%d/%m/%Y" if gran == "Tuần" else "%m/%Y"
        cd = np.c_[g["symbol"], g["t"].dt.strftime(fmt), g["n"], g["pos"], g["neg"]]
        hov = ("<b>%{customdata[0]}</b> · %{customdata[1]}<br>sentiment %{z:+.3f}<br>%{customdata[2]} tin · "
               "%{customdata[3]:.0%} tích cực · %{customdata[4]:.0%} tiêu cực<extra></extra>")

        view = st.radio("Kiểu xem", ["3D", "Bản đồ nhiệt 2D"], horizontal=True, key="v3_view",
                        help="Bản đồ nhiệt là cùng dữ liệu nhìn từ trên xuống: dễ so chính xác hơn, không cần xoay.")
        if view == "3D":
            zl = max(0.3, float(g["score"].abs().max()) * 1.1)
            fig3 = go.Figure()
            # one ribbon per ticker along time: the trend of each company reads as a line, not a cloud
            for s_, gs in g.sort_values("t").groupby("symbol"):
                fig3.add_scatter3d(x=gs["xd"], y=gs["yi"], z=gs["score"], mode="lines+markers", name=s_,
                                   customdata=cd[gs.index], hovertemplate=hov, showlegend=False,
                                   line=dict(width=4, color=gs["score"], colorscale=DIVERGING, cmin=-0.6, cmax=0.6),
                                   marker=dict(size=np.clip(2.5 + 1.6 * np.sqrt(gs["n"]), 3, 14), color=gs["score"],
                                               colorscale=DIVERGING, cmin=-0.6, cmax=0.6, opacity=.95,
                                               line=dict(width=0), showscale=s_ == order[-1],
                                               colorbar=dict(title="Sentiment", thickness=10, len=.55)))
            fig3.add_surface(x=[0, g["xd"].max()], y=[-0.5, len(order) - 0.5], z=[[0, 0], [0, 0]],
                             colorscale=[[0, "#898781"], [1, "#898781"]], opacity=.12, showscale=False,
                             hoverinfo="skip")
            fig3.update_layout(height=680, margin=dict(t=10, b=10, l=0, r=0),
                               scene=dict(xaxis=dict(title="Thời gian", tickvals=tickvals, ticktext=ticktext),
                                          yaxis=dict(title="", tickvals=list(range(len(order))), ticktext=order),
                                          zaxis=dict(title="Sentiment", range=[-zl, zl]),
                                          aspectmode="manual", aspectratio=dict(x=2.2, y=1.4, z=0.8),
                                          camera=dict(eye=dict(x=1.25, y=-2.35, z=0.85), center=dict(x=0, y=0, z=-0.25))))
            st.plotly_chart(fig3, width="stretch", key="v3_scatter")
        else:
            pv = g.pivot(index="symbol", columns="t", values="score").reindex(order)
            pn = g.pivot(index="symbol", columns="t", values="n").reindex(order).fillna(0)
            figh3 = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, customdata=pn.values,
                                         colorscale=DIVERGING, zmin=-0.6, zmax=0.6, xgap=2, ygap=2,
                                         colorbar=dict(title="Sentiment", thickness=10),
                                         hovertemplate="<b>%{y}</b> · %{x|" + fmt + "}<br>sentiment %{z:+.3f}"
                                                       "<br>%{customdata:.0f} tin<extra></extra>"))
            figh3.update_layout(height=max(320, 28 * len(order) + 90), margin=dict(t=10, b=40, l=60, r=20))
            st.plotly_chart(figh3, width="stretch", key="v3_heat")
            st.caption("Ô trống = kỳ đó không có tin (hoặc mọi tin đều bị abstain): thiếu dữ liệu, không phải 0.")

        # the table twin: what the chart says, in numbers
        summ = []
        for s_ in reversed(order):
            gs = g[g["symbol"] == s_].sort_values("t")
            summ.append({"Mã": s_, "Số tin": int(gs["n"].sum()),
                         "Sentiment TB": round(float((gs["score"] * gs["n"]).sum() / gs["n"].sum()), 3),
                         "Kỳ gần nhất": gs["t"].iloc[-1].strftime(fmt),
                         "Sentiment kỳ gần nhất": round(float(gs["score"].iloc[-1]), 3),
                         "Thay đổi so kỳ trước": round(float(gs["score"].iloc[-1] - gs["score"].iloc[-2]), 3)
                                                 if len(gs) > 1 else None})
        st.dataframe(pd.DataFrame(summ), hide_index=True, width="stretch",
                     column_config={"Sentiment TB": st.column_config.ProgressColumn(
                         min_value=-1, max_value=1, format="%+.3f")})
        st.caption("Sắp theo sentiment trung bình (cao → thấp). Sentiment đo **tin nói gì về công ty**, "
                   "không phải dự báo giá.")
