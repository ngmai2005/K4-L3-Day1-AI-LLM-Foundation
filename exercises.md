# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Qua bốn mức temperature, có thể nhận thấy rằng temperature càng cao thì câu trả lời thường càng đa dạng và sáng tạo hơn, nhưng độ ổn định có thể giảm. Ở mức thấp như 0.0, mô hình trả lời nhất quán và ít biến đổi; ở mức cao như 1.0 hoặc 1.5, nội dung có thể bất ngờ hơn nhưng đôi khi kém chính xác hoặc lan man hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đặt temperature = 0.2 hoặc khoảng 0.0–0.3 cho chatbot hỗ trợ khách hàng. Mức temperature thấp giúp câu trả lời ổn định, nhất quán và hạn chế việc mô hình trả lời lan man hoặc đưa ra thông tin không chính xác. Chatbot chăm sóc khách hàng ưu tiên độ tin cậy và tuân thủ kịch bản hơn là sự sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.000 người dùng/ngày, mỗi người gọi API 3 lần, tổng cộng có 30.000 lượt gọi/ngày. Mỗi lượt trung bình 350 token đầu ra, tổng số token đầu ra là khoảng 10,5 triệu token/ngày.

Gemini 2.5 Flash có chi phí cao hơn Gemini 2.5 Flash-Lite khoảng 6 lần cho phần token đầu ra theo mức giá tham khảo trong bài thực hành.

Flash: phù hợp với tác vụ phức tạp như phân tích dữ liệu, lập trình hoặc suy luận nhiều bước.

Flash-Lite: phù hợp với FAQ, phân loại yêu cầu và trả lời ngắn để tiết kiệm chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Persona giáo viên tiểu học tạo câu trả lời ngắn, dễ hiểu, dùng từ đơn giản và ví dụ gần gũi. Persona chuyên gia tài chính tạo câu trả lời chuyên sâu, dài hơn và sử dụng thuật ngữ như “sổ cái phân tán”, “mã hóa” và “cơ chế đồng thuận”. System prompt định hướng giọng điệu, độ sâu, từ vựng và cách model trình bày câu trả lời.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn khoảng 100 từ:
Ước lượng: 100 / 0.75 ≈ 133 token.
count_tokens() thực tế có thể khoảng 200 token.
Chênh lệch: (200 - 133) / 133 × 100 ≈ 50%.
Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì tokenizer có thể chia từ tiếng Việt có dấu thành nhiều token nhỏ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming phù hợp khi cần hiển thị kết quả ngay từng phần, giúp người dùng cảm thấy phản hồi nhanh hơn. Non-streaming phù hợp khi cần nhận toàn bộ kết quả để xử lý hoặc lưu trữ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ giữa các lần retry, giúp API có thời gian phục hồi và giảm quá tải. Nếu hàng nghìn client cùng retry sau 1 giây, chúng sẽ gửi request đồng loạt, gây quá tải tiếp tục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Bạn là trợ lý học tập CNTT thân thiện. Hãy trả lời bằng tiếng Việt, giải thích kiến thức theo từng bước, sử dụng ví dụ thực tế và giải thích các thuật ngữ chuyên môn khi xuất hiện. Nếu người dùng gặp lỗi code, hãy chỉ rõ nguyên nhân và cách sửa, không tự ý thay đổi những phần code đang hoạt động.
Việc yêu cầu “trả lời bằng tiếng Việt” giúp nội dung phù hợp với người học. Cụm “giải thích theo từng bước” giúp người mới dễ hiểu và có thể thực hành theo.
### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là chỉ lưu một số lượt hội thoại gần nhất nên có thể quên những thông tin quan trọng ở các cuộc trò chuyện trước. Cải thiện cụ thể là xây dựng bộ nhớ dài hạn bằng cách lưu lịch sử vào cơ sở dữ liệu hoặc file JSON. Khi người dùng đặt câu hỏi, chương trình sẽ tìm lại những thông tin liên quan và đưa chúng vào prompt trước khi gọi API.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
