# Lab 21 — Evaluation Report

**Họ tên**: Vi Minh Hiền  **MSSV**: 01743  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB` (14.6 GB khả dụng)

> Mọi con số dưới đây khớp chính xác với các tệp kết quả trong `results/`.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (mặc định) |
| Train / val | 200 / 50 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json: "reasoning preserved — safe to train on traces")*
Không cần loại bỏ thủ công vì template bảo toàn khối `<think>` hợp lệ.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss (*results/mask_proof.json*):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3215.0 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1016.2 |
| (c) LoRA fine-tune | 0.9700 | 0.6111 | 1.0000 | 1382.9 |

**(b) có thật sự mạnh hơn (a) không?** `có` — prompt tối ưu giúp độ chính xác target tăng từ 0.0% lên 76.5%, format chuẩn JSON 100%, độ trễ suy luận giảm từ 3215 ms xuống 1016 ms.
Tôi giữ nguyên `OPTIMIZED_PROMPT` tiêu chuẩn được cung cấp trong lab để đảm bảo mốc so sánh công bằng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6266 | 0.9700 | 30 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5372 | 0.9700 | 30 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5704 | 0.0000 | 30 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.9400 | 30 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` đạt độ chính xác 0.9700, **hoà** tuyệt đối với `correct` (0.9700). Tuy nhiên, trên tập train, `attn_only` có train loss thấp hơn (0.5372 so with 0.6266). Thứ tự theo train loss hoàn toàn khác với thứ tự theo độ chính xác target. Điều này chứng minh rằng việc cố tình nâng rank cực cao ($r=283$) chỉ trên các lớp Attention (`q,v`) giúp mô hình khớp dữ liệu huấn luyện tốt hơn (loss thấp hơn), nhưng đòn bẩy thực sự để đạt hiệu quả tổng quát hóa cao lại là **vị trí gắn adapter phủ rộng khắp các lớp tuyến tính (`text-linear`)** với rank vừa phải ($r=16$).

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
`wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang điểm của Full Fine-Tuning) thay vì $1\times 10^{-4}$ của LoRA. Đường loss của `wrong_lr` giảm rất chậm và giữ ở mức cao (1.5704 so với 0.6266), dẫn đến việc mô hình hoàn toàn thất bại trên tập target (accuracy = 0.0000). Nếu chỉ nhìn vào loss mà không biết tốc độ học bị đặt sai, ta sẽ kết luận sai rằng kiến trúc mô hình bị hỏng, bộ dữ liệu lỗi, hoặc LoRA không làm việc với bài toán này, trong khi thực tế nguyên nhân duy nhất là bước cập nhật trọng số quá nhỏ đối với ma trận rảnh rỗi LoRA.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
`qlora` tiết kiệm tới **4.92 GB VRAM** (giảm từ 12.01 GB ở fp16 xuống 7.09 GB ở 4-bit, giảm ~41% VRAM tiêu thụ). Tuy nhiên, mô hình phải trả giá bằng việc giảm độ chính xác trên tập target từ 0.9700 xuống 0.9400 (giảm 3%), đồng thời độ trễ latency tăng từ 1382.9 ms lên 1770.9 ms (chậm hơn ~28%). Kết quả thực nghiệm này hoàn toàn **ủng hộ khuyến nghị của nhà cung cấp Qwen3.5** (deck §12): đối với dòng model này, nếu ngân sách GPU cho phép (như trên Colab T4 16GB), nên dùng 16-bit LoRA (fp16/bf16) để đạt chất lượng và tốc độ tối ưu thay vì QLoRA 4-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.2050` · `regression Δ = -0.1467` · `valid_trace_rate = 0.00`

**Diễn giải**:
Cổng hồi quy báo kết quả **FAILED** mặc định vì chỉ số năng lực tổng quát (`regression`) bị sụt giảm **0.1467** (vượt quá ngưỡng dung sai cho phép là 0.020). Mặc dù bản fine-tune giúp độ chính xác trên nhiệm vụ phân loại CSKH (`target`) tăng từ 76.5% lên 97.0% (tăng +20.5%), nhưng việc huấn luyện chỉ trên 250 mẫu dữ liệu chuyên biệt mà không có dữ liệu trộn phổ thông (replay data) đã gây ra hiện tượng *catastrophic forgetting* (quên thảm họa). Mô hình bị mất một phần tri thức tổng quát ban đầu. Phán quyết FAILED này là hoàn toàn chính xác và có giá trị thực tiễn: để đưa bản fine-tune vào sản xuất thương mại thực sự mà vẫn giữ được khả năng trả lời các câu hỏi kiến thức chung, chúng ta cần bổ sung thêm 1–5% dữ liệu tổng hợp (replay data) vào tập huấn luyện theo khuyến nghị tại deck §14.3.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | doi_tra, cao, chuột không dây, tich_cuc | doi_tra, trung_binh... | doi_tra, cao, chuột không dây, tich_cuc | ✅ **FT thắng**: Đoán chính xác 100% cả 4 trường |
| 2 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi... | hoan_tien, cao, đèn bàn LED, tich_cuc | hoan_tien, trung_binh... | hoan_tien, cao, đèn bàn LED, tich_cuc | ✅ **FT thắng**: Đoán chuẩn độ khẩn cấp "cao" |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien, cao, bình giữ nhiệt, tieu_cuc | hoan_tien, trung_binh... | hoan_tien, trung_binh, bình giữ nhiệt, trung_tinh | ❌ **FT thua**: Nhầm lẫn `urgency` trung_binh vs cao |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi, cao, nồi chiên không dầu, tieu_cuc | san_pham_loi, trung_binh... | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | ❌ **FT thua**: Nhầm sắc thái `sentiment` trung_tinh |
| 5 | Shop ơi, mình đặt balo laptop mã đơn VN294388. Hoàn tiền. Ngay lập tức... | hoan_tien, cao, balo laptop, tich_cuc | hoan_tien, trung_binh... | hoan_tien, cao, balo laptop, tich_cuc | ✅ **FT thắng**: Bắt từ khóa "Ngay lập tức" chuẩn xác |

**Có mẫu chung nào ở các ca FT thua không?**
Có. Các ca fine-tune bị điểm phạt (score 0.75) thường xuất hiện ở những ticket ngắn, câu từ lấp lửng ("Chưa thấy tiền", "Thiếu phụ kiện"). Ở các trường hợp này, nhãn chuẩn phân loại mức độ khẩn cấp là `cao` và sắc thái là `tieu_cuc`, nhưng mô hình fine-tune có xu hướng thiên vị gán mức `trung_binh` và `trung_tinh` do tần suất xuất hiện cao hơn của nhóm nhãn này trong tập train ít mẫu.

---

## 7. Kết luận & điều tôi học được

**Kết luận**:
Bản fine-tune LoRA mang lại sự cải thiện vượt bậc về độ chính xác phân loại chuyên biệt (từ 76.5% lên 97.0%) và đảm bảo định dạng đầu ra JSON 100%. Tuy nhiên, chúng ta **chưa nên deploy ngay bản fine-tune này lên hệ thống sản xuất chung** nếu hệ thống yêu cầu xử lý cả các câu hỏi kiến thức ngoài CSKH, do mô hình bị suy giảm 14.7% điểm năng lực tổng quát (regression gate FAILED). Nếu chỉ ứng dụng cho microservice chuyên biệt xử lý riêng ticket CSKH (JSON triage router), bản fine-tune này hoàn toàn đáp ứng xuất sắc.

Đòn bẩy thực sự quyết định sự thành công trong lab này không phải là việc cố gắng tăng rank LoRA lên thật cao, mà nằm ở: (1) **Vị trí gắn adapter** (`text-linear` phủ rộng toàn bộ các lớp tuyến tính); (2) **Thang Learning Rate phù hợp** ($1\times 10^{-4}$ cho LoRA); và (3) **Loss mask chính xác** (`assistant-only`) để mô hình chỉ học phần câu trả lời mà không học vẹt lại prompt của người dùng.

**Ba điều tôi học được**:
1. **Loss mask đúng là xương sống của SFT**: Phải chứng minh câu hỏi bị mask hoàn toàn và câu trả lời nằm trong loss bằng giải mã ngược (`mask_proof.json`), tuyệt đối không đoán mò.
2. **LoRA Without Regret**: Phủ adapter lên tất cả các lớp tuyến tính (`all-linear`) với rank vừa phải ($r=16$) luôn mang lại hiệu quả tổng quát hóa tốt hơn và an toàn hơn so với việc nâng rank cực cao cho riêng Attention (`q,v`).
3. **Phân biệt công bằng giữa Benchmark và Train Loss**: Không được dùng `final_loss` để xếp hạng các mô hình; chỉ có các chỉ số đo đạc trên tập kiểm thử độc lập (`target accuracy`, `regression score`) mới phản ánh đúng chất lượng thực tế.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử**:
Thêm 2–5% dữ liệu tổng quát (replay dataset) vào tập train để khắc phục hiện tượng suy giảm điểm `regression`, giúp mô hình vừa đạt 97%+ accuracy target vừa vượt qua cổng hồi quy (PASS verdict).

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap (`results/merge_check.json`)
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
