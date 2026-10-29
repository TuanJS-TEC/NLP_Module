# Dự báo biến động mạnh T+2 / T+5 — thiết kế (2026-10-28)

## Ý định và lý do
Chủ dự án muốn model dự báo **khớp với giá thật** của từng mã ở T+2 và T+5. Không muốn dừng ở việc gắn nhãn
tích cực/tiêu cực/trung tính cho tin. Yêu cầu: **độ chính xác 0,65–0,75**, và model **chỉ đưa ra dự báo khi đủ tự tin**.
Chủ dự án chấp nhận model im lặng ở phần lớn các phiên.

Đã đo trước khi thiết kế (spike 2026-10-27, script tạm, không phải code dự án):
- Dự báo **chiều tăng/giảm** không đạt 0,65 ở bất kỳ độ phủ nào. Hơn/kém thị trường đạt khoảng 0,60 ở 20% số phiên.
- Dự báo **biên độ** (`absmove`) đạt 0,65–0,72 trên test khi chỉ dự báo ở 3–5% số phiên. Ngưỡng chọn trên validation.
  Số lần dự báo ít (65–140), CI khoảng ±0,09.
- Tin tức và BCTC không cộng thêm gì. Tín hiệu đến từ giá và khối lượng.

Chủ dự án đã chọn: **làm dự báo biến động mạnh**, hiển thị ở **tab mới cạnh "Sentiment vs Giá SSI"** trong `app.py`.

## Mục tiêu
- Mỗi mã, mỗi phiên, cho h ∈ {2, 5}: dự báo |lợi suất từ close t đến close t+h| có lớn hơn trung vị các mã trong
  cùng ngày hay không. Đây đúng là `trial_forecast.add_targets(..., "absmove")`. Kết quả: **Mạnh / Yếu / Không dự báo**,
  kèm xác suất.
- Chỉ dự báo khi độ tự tin |p − 0,5| vượt ngưỡng. Ngưỡng là mức có độ phủ lớn nhất mà vẫn đạt ≥ 0,65 trên
  validation, với ít nhất 50 lần dự báo. Nếu không có mức nào đạt thì không dự báo, và báo rõ điều đó.
- Đánh giá trung thực: walk-forward theo `walk_forward.FOLDS` (A/B/C). Mỗi fold báo độ chính xác test, độ phủ,
  số lần dự báo, CI Wilson 95%.
- Có **mốc so sánh**: quy tắc "vol10 cao nhất → Mạnh, thấp nhất → Yếu", cùng độ phủ với model. Model không hơn
  mốc thì ghi rõ trong report.
- Dự báo hằng ngày tự cập nhật từ dữ liệu cron đã thu (không gửi thêm request nào), hiển thị trong tab mới.

## Không làm
- Dự báo chiều tăng/giảm, giao dịch tự động, khuyến nghị mua/bán.
- Feature NLP/BCTC trong model mặc định. Vẫn có cờ để thử lại, nhưng mặc định tắt vì spike không thấy lợi ích.

## Không được xảy ra (dù chạy được vẫn là thất bại)
- Con số 0,65 đến từ việc chọn ngưỡng hoặc model trên chính tập test, hoặc từ look-ahead. Mọi feature tại phiên t
  chỉ được dùng dữ liệu đến close t. Train và validation phải purge h phiên, giống `walk_forward._split`.
- Tab mới hiển thị "độ chính xác 0,7" mà không kèm độ phủ và số lần dự báo.

## Ràng buộc
- **Dữ liệu giá lịch sử:** `data/stock/{SYM}_stock.csv`, 15 mã, cột `time,open,high,low,close,volume`, đơn vị
  nghìn VND, có vẻ đã điều chỉnh cổ tức, kết thúc 2026-10-21.
- **Giá live:** bảng `market_quote` (cron mỗi 15 phút, từ 2026-10-26). Có `matched_price, highest, lowest,
  volume, ref_price, session, trading_date`, đơn vị VND, **chưa điều chỉnh**, không có giá mở cửa.
  Mỗi phiên lấy bản ghi cuối cùng của `trading_date` có `fetched_at >= 08:00 UTC` (sau giờ đóng cửa 15:00) làm OHLCV
  của phiên đó: close = `matched_price`, chia cho 1000 để về nghìn VND. Phiên chưa có bản ghi sau 15:00 thì chưa tính.
- **Khoảng trống:** 22–23/10/2026 không có trong file, cũng không có trong `market_quote`. Giữ nguyên là phiên thiếu,
  rolling tính trên các hàng có. Ghi chú lại; không đi tìm nguồn khác.
- **Lệch do cổ tức:** ngày chia cổ tức, nối giá chưa điều chỉnh với giá đã điều chỉnh sẽ sinh một lợi suất giả.
  Chấp nhận, và ghi `# ponytail:` (nâng cấp: dùng `ref_price` của SSI làm close hôm trước khi |ret1| vượt biên độ ±7%).
- Không dependency mới: sklearn, pandas, streamlit đã có sẵn.
- Quy tắc nguồn dữ liệu (`docs/source_terms.md`) giữ nguyên. Phần này không gửi thêm request nào.

## Phần đã quyết
- **Universe:** cả 15 mã trong `data/stock/` (không cần NLP nên không bị giới hạn ở 10 mã).
- **Feature giá** (đã dùng trong spike): `ret1, ret5, ret20, vol10, vol20, ma20` (close/MA20 − 1),
  `range` ((high−low)/close), `vz` (log volume so với trung bình 20 phiên), `mret1, mret5` (trung bình các mã).
  Feature thị trường tính cross-section trên 15 mã.
- **Model:** chọn trên validation giữa logistic (StandardScaler + LR) và HistGradientBoosting
  (`max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=100`). Sau đó train lại trên train+val để chấm test.
- **Model chạy live:** train trên toàn bộ lịch sử có nhãn. Ngưỡng lấy từ fold mới nhất (C). Lưu bằng joblib như
  `baseline.joblib`.
- **Hiển thị:** tab mới tên "Dự báo biến động", đặt ngay sau "Sentiment vs Giá SSI".

## Để builder tự quyết
- Tên module, ví dụ `nlp_module/volcast.py`, theo khuôn `trial_forecast.py` (có `main()` riêng). Có thể dùng lại
  `add_targets`, `walk_forward._split`, `FOLDS`.
- Nội dung tab, ít nhất gồm:
  - Bảng các mã × {T+2, T+5}: Mạnh/Yếu/—, xác suất, phiên làm căn cứ.
  - Hộp "độ tin cậy đã đo" với độ chính xác, độ phủ, n, CI của từng fold và của mốc so sánh.
  - Ghi chú: dự báo biên độ, không phải chiều tăng/giảm.
- Có ghi dự báo hằng ngày vào DB để sau này chấm điểm live hay không. Gợi ý: một bảng `vol_forecast` để sau đo được
  độ chính xác thật trên SSI.
- Có cập nhật `scripts/live_update.sh` để tự chạy sau phiên hay không.
- Một test nhỏ cho phần không look-ahead (feature tại t không đổi khi giá sau t thay đổi) và cho logic chọn ngưỡng.

## Kết quả phải ghi lại
Bảng walk-forward (fold × h: độ chính xác, độ phủ, n, CI, mốc so sánh) được thêm vào `docs/model_report.md`,
kể cả khi không đạt 0,65.
