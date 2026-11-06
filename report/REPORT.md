# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## Phần trả lời:
## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Với `sample_id = traffic`,  `rank = 1`, `class_id = 468`, `class_name = cab`, `score = 0.510914`, `taxonomy_name = ImageNet-1K`
2. Lúc đầu mình nghĩ `cab` chỉ một vật thể vì đó là tên một loại xe. Khi xem lại, mình hiểu đây là dự đoán cho cả ảnh vì model nhận nguyên ảnh rồi xếp hạng các lớp. `rank = 1` nghĩa là lớp `cab` có điểm cao nhất. Ảnh có cả xe buýt lẫn ô tô nên từ kết quả này mình chưa xác định được model dựa vào chiếc xe nào. 
3. Danh sách tên lớp do Model học từ các mẫu huấn luyện có sẵn trong taxonomy đi kèm checkpoint rồi xếp hạng các lớp trong danh sách đó khi gặp ảnh mới.
4. class_id chi biết mã lớp, class_name cho biết tên lớp, còn taxonomy_name cho biết phải tra mã ở bảng nào. Nếu chỉ ghi mỗi class_name, sẽ khó nhận biết dữ liệu thuộc lớp dữ liệu nào, tên lớp tên bảng dữ liệu. Ghi đủ cả ba giúp người đọc hiểu kết quả, phần mềm tra đúng lớp và người kiểm tra phát hiện trường hợp mã và tên lớp bị ghi lệch nhau. 
5. Nếu mỗi ảnh chỉ được gán một lớp, guideline cần nói rõ chọn theo chủ thể nào. Nếu lấy phương tiện nổi bật nhất làm chủ thể, cần nói thêm cách xử lý khi hai loại xe đều nổi bật. Cái nào chưa có quy tắc thì đánh dấu để hỏi.
6. `score` cho mình biết model chấm lớp đó cao hay thấp. Ground truth là nhãn người gán dựa vào ảnh và guideline, sau đó được kiểm tra. Dù điểm cao hay không thì, Model vẫn có thể đoán sai.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chọn box `person` ở mép trái. Bản ghi: `score = 0.610931`, `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]`, `bbox_width = 61.44`, `bbox_height = 47.59`. 
2. Góc trên trái của hộp ở `(0.12, 263.18)`, gần sát mép trái ảnh. Góc dưới phải ở `(61.55, 310.78)`. Box rộng `61.44` px và cao `47.59` px.
3. Trong ảnh `kitchen`, ngưỡng `0.20` có 17 vật thể và `0.60` có 6 vật thể.
4. Khi hạ ngưỡng, người kiểm tra có thêm hộp để xem model có bỏ sót vật ở ngưỡng cao hay không. Họ cũng cần lọc các hộp sai lớp, trùng nhau hoặc bao lệch.
5. Nếu guideline yêu cầu gán theo phần nhìn thấy, mình sẽ vẽ một hộp chữ nhật sát phần nhìn thấy của mỗi vật thể. Với vật bị cắt ở mép, box dừng trong khung ảnh.
6. Mép trái ảnh chỉ thấy bàn tay và một phần cánh tay. Mình sẽ đánh dấu ca này và hỏi Lab Coach nếu guideline chưa nói rõ phần tay nhìn thấy có đủ để gán lớp vật thể hay không.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Trong ảnh `kitchen`. Bản ghi: `instance_id = kitchen-001`, `class_name = person`, `score = 0.899318`, `polygon_point_count = 348`. Năm điểm đầu của `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`.
2. Polygon cho mình thấy hình dáng cụ thể hơn. Trên ảnh, mask đi theo vai và hai chân của người, còn hộp chữ nhật bao luôn khoảng trống giữa hai chân.
3. instance_id dùng để phân biệt từng vật thể mà model phát hiện trong ảnh, kiểu nó gán `id` vật thể và giúp mình tìm đúng mask của vật thể đó.
4. Nếu guideline yêu cầu mask phần nhìn thấy, mình sẽ đặt điểm theo đường viền vật thể, thêm điểm ở chỗ đổi hướng và tránh tô sang nền hoặc đồ vật nằm phía trước.
5. Trong ảnh `kitchen`, mình thấy miếng bột cán che một phần mặt bàn. Khi so với ảnh phân đoạn, mình thấy mask của bàn phủ lên cả miếng bột. Nếu áp dụng quy tắc chỉ vẽ phần nhìn thấy, mình sẽ dùng guideline và để vùng miếng bột nằm ngoài mask bàn. Nếu guideline chưa nói rõ cách xử lý đồ vật đặt trên bàn, mình sẽ hỏi Lab Coach để phần gán nhã được nhất quán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Nhãn cần tạo | Chỗ khó trong ảnh | Người gán nhãn làm gì? | Người kiểm tra xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh. | Ảnh `traffic` có cả ô tô và xe buýt, nên khó chọn một nhãn. | Chọn loại xe chính theo quy tắc của bài. | Xem cách chọn nhãn có đúng quy tắc không. |
| Phát hiện vật thể | Một tên lớp và một khung cho mỗi vật thể. | Mép trái ảnh `kitchen` chỉ thấy cánh tay của một người. | Nếu quy tắc cho phép gán nhãn, vẽ khung quanh phần nhìn thấy. | Xem trường hợp này có cần gán nhãn và khung có vừa không. |
| Phân đoạn | Một tên lớp và một vùng tô theo hình dáng của mỗi vật thể. | Trong ảnh `kitchen`, vùng tô của bàn phủ lên cả miếng bột. | Nếu chỉ lấy phần bàn nhìn thấy, để miếng bột ngoài vùng tô. | Xem vùng tô có lấn sang miếng bột không. |

## 5. An toàn dữ liệu

1. Mình chỉ dùng ảnh COCO công khai có sẵn trong notebook, giữ file ghi nguồn ảnh và tránh đưa dữ liệu cá nhân hoặc nội bộ vào bài nộp.
2. Nếu gặp dữ liệu ngoài phạm vi, mình sẽ dừng xử lý và báo cho Lab Coach trước khi chia sẻ hay upload.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS

