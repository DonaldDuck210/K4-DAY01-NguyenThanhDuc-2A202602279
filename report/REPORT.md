# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/9/2026**

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
- Record này mô tả toàn ảnh như thế nào? - Record mô tả ảnh trong tập lớp của taxonomy ImageNet-1K, là lớp rank cao nhất khi xét class_name là cab với class_id=468, với score là 0.510915
- Ai định nghĩa class list mà checkpoint có thể dự đoán? - Biến taxonomy_name
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? - 3 biến này giúp xác định đúng vật với ID, giúp con người nhận biết vật qua tên lớp và ID của vật nằm ở taxonomy nào.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? - Phải xác định labeling guideline trước khi đánh giá model
- Vì sao model score không phải ground truth? - Vì model score chỉ là thông số đánh giá model còn ground truth là kết quả, thực tế đã được kiểm chứng

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
- Diễn giải vị trí box bằng lời: Ví dụ với dining table
bbox = [0.0, 318.65, 478.73, 638.83] thì có thể nói "Một bàn ăn rất lớn chiếm gần như toàn bộ chiều ngang và phần dưới của ảnh; box chạm cả mép trái và gần chạm mép dưới."
- So sánh số prediction ở hai threshold: Số prediction giảm từ 53 xuống 33 với threshold lần lượt = 0.35 và 0.5
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Nếu thay đổi threshold = 0.5 sẽ giảm workload và giảm false-positive candidates, nhưng có nguy cơ bỏ sót các object có score thấp 
- Đề xuất một quy tắc box chặt: Bounding box phải là hình chữ nhật nhỏ nhất bao trọn toàn bộ phần object nhìn thấy, bám sát biên ngoài của object và không bao gồm background không cần thiết.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Nếu object bị cắt bởi mép ảnh nhưng phần nhìn thấy đủ để xác định object, annotate đúng phần nhìn thấy và cho phép box chạm mép ảnh. Không kéo box ra ngoài ảnh.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box? Polygon so với box mô tả đường biên của đối tượng chi tiết hơn, nên có thể biểu diễn hình dạng không phải hình chữ nhật và loại bỏ phần background nằm bên trong box.
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id dùng để phân biệt từng đối tượng cụ thể trong ảnh, kể cả khi chúng cùng class_name.
- Đề xuất một quy tắc biên mask: Mask phải bao phủ toàn bộ phần nhìn thấy của đối tượng và bám sát biên ngoài thực tế, đồng thời không bao gồm background hoặc vùng thuộc đối tượng khác.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Chỉ annotate phần đối tượng có thể xác định một cách đáng tin cậy từ hình ảnh.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn (class) cho mỗi ảnh | Ảnh có thể chứa nhiều đối tượng nhưng không rõ class chính | Chọn một class phù hợp nhất theo guideline | Kiểm tra nhãn có đúng với nội dung ảnh không |
| Phát hiện vật thể | Mỗi object = 1 bounding box dạng xyxy = [x1, y1, x2, y2] + class | Box quá rộng/hẹp, chứa background, cắt mất phần object; object bị crop, occlusion, blur hoặc chạm biên ảnh | Annotate từng object riêng biệt, đặt box sát nhất với phần object nhìn thấy | Kiểm tra đúng class, đủ số object và vị trí/kích thước box |
| Instance segmentation | Mỗi object instance = 1 polygon/mask (polygon_xy) + class | Biên object khó xác định do blur, occlusion, tiếp xúc/chồng lấn, object có hình dạng phức tạp hoặc rất nhỏ | Annotate từng instance riêng biệt, mask phải bao phủ toàn bộ phần nhìn thấy và bám sát biên thực tế | Kiểm tra đúng class, đúng số instance và chất lượng biên mask |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Sử dụng dữ liệu đúng phạm vi cho phép
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Người có quyền hành cao nhất trong dự án

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
