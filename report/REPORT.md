# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

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

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
1.1. classification_predictions
  {
    "taxonomy_name": "ImageNet-1K",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },

1.2. classification_predictions
{
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.912558,
  },

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record này nói về vật thể riêng bởi vì trong này có class_name và class_id dùng để chỉ vật thể trong ảnh


3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Classlist không phải do model nghĩ ra mà do taxonomy/checkpoint bởi vì model không thể nào suy luận được ra vật thể nếu không có data dùng để huấn luyện 

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

class_id : dùng để định danh lớp
class_name: dùng để biết tên lớp
taxonomy_name: dùng để phân loại vật thể

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

Guidline cần nói rõ , tên lớp, id của lớp , tên phân loại và tên lớp

6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ là là các chỉ số định lượng dùng để đo lường, đánh giá hiệu suất, độ chính xác 

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

{
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  },


2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
góc trên-trái ở tọa độ (385.33,69.24), 
góc dưới-phải ở tọa độ (498.92,348.92), 
 rộng 113.58 cao 279.68
 
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
0.2: 17 0.35: 11 0.6: 6

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng model giữ lại nhiều hộp không tin cậy -> người review cần loại bỏ chúng

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- gán nhãn đúng chiều cao,rộng và chính xác các vật thể đã tồn tại nhãn được phân công

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

Vật che khuất là cốc bạc 
Nếu guideline đã có quy tắc rõ thì áp dụng. Nếu chưa rõ có gán nhãn vật đó không hoặc vẽ box thế nào, hãy hỏi Lab Coach (escalation) vì tự đặt quy tắc có thể khiến cách gán nhãn không thống nhất. Sau khi được thống nhất, bổ sung quy tắc vào guideline.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm 
polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

{    "instance_id": "traffic-001", "class_name": "bus", "score": 0.925745, "polygon_point_count": 120, "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],]}
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Nó giúp biết pixel nào thuộc vật thể và pixel nào thuộc nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

Đây là mã để phân biệt từng đối tượng cụ thể trong ảnh

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

Vẽ mask sát theo phần viền nhìn thấy của vật thể; tại chỗ bị vật khác che, dừng ở đường che khuất và chỉ gán nhãn phần nhìn thấy, trừ khi guideline có quy định khác

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
cái cốc bạc
Nếu guideline chưa nói rõ cách xử lý chỗ biên mờ, hai vật chạm nhau hoặc bị che, hãy escalate lên Lab Coach để thống nhất; sau đó bổ sung quyết định vào guideline để mọi người gán nhãn nhất quán

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Một hoặc nhiều nhãn cho toàn ảnh|Vật thể nhỏ, bị che hoặc cắt mép có được tính không; dễ nhầm nhãn; thiếu hoặc thừa nhãn  | Gán nhãn cho ảnh theo taxonomy và quy tắc trong guideline; hỏi Lab Coach nếu trường hợp chưa được quy định |Nhãn có đúng nội dung ảnh không; có bỏ sót hoặc gán thừa lớp không; cách áp dụng taxonomy có nhất quán không  |
| Phát hiện vật thể |Mỗi vật thể có class và bounding box dạng [x_max,y_max,x_min,y_min]  |Box quá rộng/hẹp; tọa độ hoặc kích thước không khớp; vật bị che, cắt mép; box trùng hoặc dự đoán sai lớp trong JSON  | Vẽ box theo quy tắc của dự án, gán đúng lớp, kiểm tra các trường như class_name, score, bbox_xyxy, bbox_width, bbox_height | Box có thừa hay thiếu không ,trung thông tin JSON ko. |
| Instance segmentation | mỗi vật có class và instance _id, polygon riêng của nó  | Biên mờ, hai vật chạm nhau, vật bị che; polygon lệch khỏi viền |Vẽ polygon theo phần viền nhìn thấy; tạo ID riêng cho từng đối tượng  | Polygon, class , instance_id|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Không cho người không có thẩm định hay liên quan sử dụng dữ liệu 
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
-Báo cho người giao dữ liệu

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
