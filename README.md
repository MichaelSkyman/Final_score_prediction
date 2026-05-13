# Final score prediction

Chương trình dự đoán điểm cuối kỳ của một môn học dựa trên điểm giữa kỳ trong file `data/TRAIN2.xlsx`.

## Phương pháp

Bài toán có một biến đầu vào (`midterm`) và một biến mục tiêu (`final`), nên repo dùng hồi quy tuyến tính một biến nhưng cài đặt bằng **TensorFlow computational graph**:

```text
z = (midterm - mean(midterm)) / std(midterm)
final_pred = b + w * z
loss = mean((final_pred - final)^2)
```

Trong code, graph được xây bằng TensorFlow/Keras:

```text
Input(midterm) -> Normalization -> Dense(1) -> y_hat -> MSE loss
```

Hai tham số `w` và `b` của lớp `Dense(1)` được học bằng **Batch Gradient Descent**. Ở mỗi epoch, chương trình tính loss trên toàn bộ tập train, TensorFlow tự động tính gradient bằng `GradientTape`, rồi SGD cập nhật tham số một lần:

```text
dL/dw = 2/n * sum((y_hat_i - y_i) * z_i)
dL/db = 2/n * sum(y_hat_i - y_i)
w = w - alpha * dL/dw
b = b - alpha * dL/db
```

Sau khi học xong, công thức được đổi lại về thang điểm gốc để dễ dùng:

```text
final_pred = b0 + b1 * midterm
```

Điểm dự báo cuối cùng được giới hạn trong thang điểm hợp lệ `[0, 10]`.

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy chương trình

Huấn luyện mô hình, sinh chỉ số đánh giá và biểu đồ:

```bash
python src/final_score_prediction.py
```

Dự đoán cho một điểm giữa kỳ cụ thể:

```bash
python src/final_score_prediction.py --midterm 7.5
```

## Kết quả sinh ra

Sau khi chạy, thư mục `outputs/` sẽ có:

- `metrics.json`: hệ số mô hình và các chỉ số MAE, MSE, RMSE, R2.
- `test_predictions.csv`: dự đoán trên tập kiểm tra.
- `regression_plot.png`: đồ thị dữ liệu và đường hồi quy.
- `residual_plot.png`: đồ thị sai số.
- `loss_plot.png`: đồ thị loss trong quá trình huấn luyện TensorFlow graph.
- `computation_graph.png`: đồ thị tính toán của mô hình.
