# Dự báo biến động mạnh T+2/T+5 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mỗi phiên, với mỗi mã trong 15 mã, đưa ra dự báo "Mạnh / Yếu / —" cho biên độ |lợi suất| ở T+2 và T+5.
Chỉ dự báo khi model đủ tự tin. Độ chính xác được đo trung thực bằng walk-forward. Kết quả hiện ở một tab mới trong
`app.py`.

**Architecture:** Một module `nlp_module/volcast.py`, theo khuôn của `trial_forecast.py` (có `main()` riêng), gồm:
nạp OHLCV (file của bạn + SSI cuối phiên) → feature giá → walk-forward (train/val/test + chọn ngưỡng) → bundle cho
live (joblib) → ghi dự báo vào bảng `vol_forecast`. App chỉ đọc bảng đó và đọc report JSON.

**Tech Stack:** Python 3.14, pandas, scikit-learn (LogisticRegression, HistGradientBoostingClassifier), joblib,
streamlit. Không thêm dependency mới.

**Spec:** `docs/superpowers/specs/2026-10-28-volatility-forecast-design.md`

## Global Constraints
- Mục tiêu độ chính xác: ngưỡng chọn trên **validation** là `min_acc = 0.65`, `min_n = 50`. Test chỉ chấm một lần.
- Feature tại phiên t chỉ dùng dữ liệu đến close t. Purge h phiên trước mỗi ranh giới, tái dùng `walk_forward._split`.
- Fold: `walk_forward.FOLDS` (A/B/C). Target: `trial_forecast.add_targets(df, h, "absmove")`. h ∈ (2, 5).
- Đơn vị giá: nghìn VND (giá SSI chia 1000). Bản ghi SSI của một phiên chỉ được dùng khi `fetched_at >= 08:00 UTC`
  của `trading_date` đó.
- Không gửi request mới. Không dependency mới. Repo không dùng git, nên bỏ các bước commit.
- Mỗi con số độ chính xác hiển thị trên UI phải kèm độ phủ và n.

## Review Focus
1. Phiên SSI mới chỉ có bản ghi trong giờ giao dịch (chưa có bản ghi sau 15:00): phải bỏ, không được coi là close.
   Test nằm ở Task 1.
2. Mã có trong file nhưng ngày cuối chưa có bản ghi SSI: vẫn dự báo, nhưng từ phiên cuối cùng có dữ liệu, và UI ghi
   rõ "phiên căn cứ". Test nằm ở Task 3.
3. Fold không có ngưỡng nào đạt ≥0,65 trên validation (`pick_threshold` trả None): report ghi "không dự báo", live trả
   toàn "—" chứ không crash. Test nằm ở Task 2.
4. Giá sau phiên t thay đổi thì feature tại t không được đổi (không look-ahead). Test nằm ở Task 1.
5. App mở khi chưa có `data/models/volcast.joblib` hoặc bảng `vol_forecast` còn rỗng: hiện hướng dẫn chạy lệnh,
   không stack trace. Kiểm tra thủ công ở Task 4.

---

### Task 1: Dữ liệu OHLCV + feature

**Files:**
- Create: `nlp_module/volcast.py`
- Test: `tests/test_volcast.py`

**Interfaces:**
- Produces:
  - `FEATS = ["ret1","ret5","ret20","vol10","vol20","ma20","range","vz","mret1","mret5"]`
  - `load_history(stock_dir="data/stock") -> pd.DataFrame`, cột `date` (Timestamp chuẩn hoá về 00:00, naive),
    `symbol, close, high, low, volume`
  - `live_bars(c) -> pd.DataFrame`, cùng cột: bản ghi cuối mỗi (symbol, trading_date) có `fetched_at` ≥ 08:00 UTC của
    ngày đó và `matched=1`. `close=matched_price/1000`, `high=highest/1000`, `low=lowest/1000`.
  - `bars(c, stock_dir="data/stock") -> pd.DataFrame`: history + live. Trùng (symbol, date) thì giữ bản trong history.
    Sắp theo (symbol, date).
  - `add_features(df) -> pd.DataFrame`: thêm FEATS, định nghĩa giống spike (`ma20 = close/rolling20 − 1`,
    `range = (high−low)/close`, `vz = log1p(volume) − rolling20 mean of log1p(volume)`, `mret*` = trung bình
    cross-section theo date).

- [ ] **Step 1: Viết test fail**
  - `test_features_do_not_see_the_future`: 2 mã × 60 phiên tổng hợp. Sửa close của các phiên > 40 thì
    `add_features(...)` tại các phiên ≤ 40 phải bằng hệt kết quả trước khi sửa, `pd.testing.assert_frame_equal` trên FEATS.
  - `test_live_bars_use_only_post_close_reading`: DB `:memory:` qua `storage.connect(":memory:")`. Chèn 3 dòng
    `market_quote` cho FPT ngày 20261028: 03:00Z (55400), 08:05Z (55700), 09:40Z (55800). Chèn thêm 1 dòng VCB chỉ có
    03:00Z. Kỳ vọng `live_bars` có đúng 1 hàng FPT với close 55.8, không có VCB.
- [ ] **Step 2:** `.venv/bin/python -m pytest -q tests/test_volcast.py`, kỳ vọng FAIL (ImportError).
- [ ] **Step 3:** Viết các hàm ở trên.
- [ ] **Step 4:** Chạy lại, kỳ vọng PASS.

### Task 2: Walk-forward + chọn ngưỡng + mốc so sánh

**Files:** Modify `nlp_module/volcast.py`, `tests/test_volcast.py`

**Interfaces:**
- Consumes: `add_features`, `FEATS`; `walk_forward.FOLDS`, `walk_forward._split`; `trial_forecast.add_targets`
- Produces:
  - `pick_threshold(conf: np.ndarray, ok: np.ndarray, min_acc=0.65, min_n=50) -> float | None`: trả độ tự tin nhỏ
    nhất c sao cho trong các hàng có conf ≥ c thì `ok.mean() >= min_acc` và `n >= min_n`. Trong các mức đạt, chọn mức
    có độ phủ lớn nhất. Không có mức nào đạt thì trả None.
  - `MODELS: dict[str, callable]` gồm `"logit"` (StandardScaler + LR C=0.1) và `"gbm"` (HistGB depth 3, lr 0.05,
    200 iter, min_samples_leaf 100, random_state 0)
  - `walk(df, h) -> list[dict]`, mỗi fold một dict gồm `fold, h, model, threshold, val_acc, val_cov, test_acc,
    test_cov, test_n, ci_lo, ci_hi, rule_acc`. Model là cái có `val_acc` cao nhất ở ngưỡng của nó; model không có ngưỡng
    thì bị loại. Nếu cả hai đều None thì vẫn trả dict với `model=None` và các số NaN.
    `rule_acc` = độ chính xác trên test của quy tắc "vol10 xếp hạng cross-section theo ngày, |rank−0,5| cao nhất →
    vol10 cao thì Mạnh, thấp thì Yếu", lấy cùng số lần dự báo `test_n`.
  - `wilson(k, n) -> (lo, hi)`
- [ ] **Step 1: Viết test fail**
  - `test_pick_threshold`: conf gồm 4 nhóm 0.4/0.3/0.2/0.1, mỗi nhóm 25 hàng. ok: nhóm 0.4 và 0.3 đều True; nhóm 0.2
    có 10/25 True; nhóm 0.1 đều False. Khi đó top 75 có acc 60/75 = 0,8 (đạt), top 100 có acc 0,6 (không đạt), nên
    `pick_threshold(conf, ok) == 0.2`. Thêm hai case: `min_n=80` → None, và `ok` toàn False → None.
  - `test_walk_reports_none_when_unreachable`: dữ liệu ngẫu nhiên (target không có tín hiệu), `min_acc` mặc định.
    `walk` không raise. Fold nào `model is None` thì `test_n` là 0 hoặc NaN.
- [ ] **Step 2:** Chạy, kỳ vọng FAIL.
- [ ] **Step 3:** Viết các hàm. Với mỗi fold: fit trên `tr`, chấm `va`, chọn ngưỡng; fit lại trên `trva`, chấm `te`
  với ngưỡng đó.
- [ ] **Step 4:** Chạy, kỳ vọng PASS.

### Task 3: Bundle live, bảng `vol_forecast`, CLI, cron

**Files:** Modify `nlp_module/volcast.py`, `nlp_module/storage.py` (thêm `CREATE TABLE IF NOT EXISTS` trong
`connect()`), `nlp_module/schema.sql`, `scripts/live_update.sh`, `tests/test_volcast.py`

**Interfaces:**
- Produces:
  - Bảng `vol_forecast(symbol, base_date TEXT 'YYYY-MM-DD', h INTEGER, p_strong REAL, call TEXT NULL, model_version,
    created_at, PRIMARY KEY(symbol, base_date, h, model_version))`. `call` ∈ {'strong','weak',NULL}.
  - `VERSION = "volcast-v1"`; `BUNDLE = "data/models/volcast.joblib"`; `REPORT = "data/volcast_report.json"`
  - `fit_live(df, h, fold_row) -> dict(model, threshold, kind)`: train model kind của fold C trên toàn bộ hàng có nhãn.
  - `predict_latest(df, bundle) -> pd.DataFrame` gồm `symbol, base_date, h, p_strong, call`: lấy phiên cuối cùng có
    feature của **từng mã**.
  - `main(argv)`, subcommand:
    - `eval`: chạy walk cho h=2,5, in bảng, ghi REPORT
    - `train`: eval + fit_live, ghi BUNDLE
    - `predict`: đọc BUNDLE, ghi vào `vol_forecast` bằng INSERT OR REPLACE
- [ ] **Step 1: Viết test fail**
  - `test_predict_latest_uses_each_symbols_own_last_day`: AAA có dữ liệu đến 28/10, BBB chỉ đến 27/10. Kỳ vọng
    `base_date` lần lượt là '2026-10-28' và '2026-10-27'.
  - `test_predict_latest_threshold_none_gives_no_call`: bundle có threshold None thì mọi `call` đều None.
- [ ] **Step 2:** Chạy, kỳ vọng FAIL.
- [ ] **Step 3:** Viết code. Trong `scripts/live_update.sh`, sau bước `quotes` và trong cùng nhánh giờ giao dịch, thêm
  điều kiện riêng: chỉ chạy khi `hm >= 1505`. Lệnh là `.venv/bin/python -m nlp_module.volcast predict || echo "volcast failed"`.
- [ ] **Step 4:** Chạy cả suite `.venv/bin/python -m pytest -q tests`, kỳ vọng PASS hết.
- [ ] **Step 5:** `.venv/bin/python -m nlp_module.volcast train && .venv/bin/python -m nlp_module.volcast predict`,
  rồi kiểm tra `SELECT count(*) FROM vol_forecast` = 30 (15 mã × 2 h).

### Task 4: Tab "Dự báo biến động" trong app

**Files:** Modify `app.py`. Thêm tab vào `st.tabs(...)` ở dòng ~352, ngay sau "Sentiment vs Giá SSI"; thêm khối
`with tab_vol:`.

**Interfaces:** Consumes bảng `vol_forecast` (đọc read-only, `@st.cache_data(ttl=300)`, giống `db_quotes`) và
`data/volcast_report.json`.

- [ ] **Step 1:** Phần đầu tab: tiêu đề, caption "dự báo **biên độ** (biến động mạnh hơn trung vị 15 mã), không phải
  chiều tăng/giảm".
- [ ] **Step 2:** Bảng các mã × T+2/T+5. Ô hiển thị "Mạnh"/"Yếu"/"—" kèm `p_strong`. Thêm cột "Phiên căn cứ". Lấy bản
  ghi mới nhất của mỗi (symbol, h).
- [ ] **Step 3:** Hộp "Độ tin cậy đã đo" lấy từ REPORT: bảng fold × h gồm test_acc, test_cov, n, CI, rule_acc. Nếu
  `test_acc < 0.65` hoặc fold không có ngưỡng thì tô cảnh báo.
- [ ] **Step 4:** Khi thiếu BUNDLE, REPORT hoặc bảng còn rỗng: `st.info` hướng dẫn lệnh `python -m nlp_module.volcast
  train && ... predict`.
- [ ] **Step 5:** Kiểm tra: `.venv/bin/python -m py_compile app.py`. Chạy `streamlit run app.py --server.headless true`
  rồi `curl` trang chủ trả 200; mở tab bằng tay nếu có trình duyệt.

### Task 5: Ghi kết quả

**Files:** `docs/model_report.md`, `README.md`
- [ ] **Step 1:** Thêm mục "Volatility forecast (volcast-v1)" vào model_report, chép bảng từ REPORT. Ghi rõ fold nào
  không đạt 0,65, và so với rule_acc thì model hơn hay kém. Hạn chế: SSI chưa điều chỉnh cổ tức, thiếu dữ liệu 22–23/10.
- [ ] **Step 2:** README: thêm 1 dòng lệnh `volcast train / predict`, và nhắc tab mới.
