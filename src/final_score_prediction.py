from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "TRAIN2.xlsx"
OUTPUT_DIR = ROOT / "outputs"


@dataclass(frozen=True)
class LinearModel:
    intercept: float
    slope: float
    graph_bias: float
    graph_weight: float
    feature_mean: float
    feature_std: float
    losses: list[float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * x


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    expected = {"midterm", "final"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df = df[["midterm", "final"]].copy()
    df["midterm"] = pd.to_numeric(df["midterm"], errors="coerce")
    df["final"] = pd.to_numeric(df["final"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least two valid rows to fit a linear model.")
    return df


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    test_size = max(1, round(len(df) * test_ratio))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def fit_tensorflow_graph(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 160,
) -> LinearModel:
    tf.keras.utils.set_random_seed(42)

    x_train = x.astype("float32").reshape(-1, 1)
    y_train = y.astype("float32").reshape(-1, 1)
    feature_std = float(np.std(x_train))
    if feature_std == 0:
        raise ValueError("Midterm scores have zero variance; cannot fit a slope.")

    normalizer = tf.keras.layers.Normalization(axis=None, name="normalize_midterm")
    normalizer.adapt(x_train)

    graph = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), name="midterm"),
            normalizer,
            tf.keras.layers.Dense(1, name="linear_output"),
        ],
        name="final_score_tensorflow_graph",
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    x_tensor = tf.convert_to_tensor(x_train)
    y_tensor = tf.convert_to_tensor(y_train)
    losses: list[float] = []

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            y_hat = graph(x_tensor, training=True)
            loss = loss_fn(y_tensor, y_hat)
        gradients = tape.gradient(loss, graph.trainable_variables)
        optimizer.apply_gradients(zip(gradients, graph.trainable_variables))
        losses.append(float(loss.numpy()))

    dense = graph.get_layer("linear_output")
    kernel, bias_array = dense.get_weights()
    weight = float(kernel[0, 0])
    bias = float(bias_array[0])
    feature_mean = float(np.ravel(normalizer.mean.numpy())[0])
    feature_std = float(np.sqrt(np.ravel(normalizer.variance.numpy())[0]))

    slope = weight / feature_std
    intercept = bias - (weight * feature_mean / feature_std)
    return LinearModel(
        intercept=float(intercept),
        slope=float(slope),
        graph_bias=float(bias),
        graph_weight=float(weight),
        feature_mean=feature_mean,
        feature_std=feature_std,
        losses=losses,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_true - y_pred
    mse = float(np.mean(errors**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(errors)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def clip_scores(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, 10)


def load_plot_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf") if bold else Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf") if bold else Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf") if bold else Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "#111827",
) -> None:
    lines = text.splitlines()
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_height = sum(line_heights) + (len(lines) - 1) * 5
    y = xy[1] - total_height / 2
    for line, width, height in zip(lines, line_widths, line_heights):
        draw.text((xy[0] - width / 2, y), line, font=font, fill=fill)
        y += height + 5


def draw_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    x_min: float = 0,
    x_max: float = 10,
    y_min: float = 0,
    y_max: float = 10,
) -> tuple:
    left, top, right, bottom = plot
    title_font = load_plot_font(24, bold=True)
    label_font = load_plot_font(17)
    tick_font = load_plot_font(14)

    draw_centered_text(draw, ((left + right) / 2, 32), title, title_font)
    draw.line((left, bottom, right, bottom), fill="#111827", width=2)
    draw.line((left, top, left, bottom), fill="#111827", width=2)

    for i in range(6):
        x_val = x_min + (x_max - x_min) * i / 5
        px = left + (right - left) * i / 5
        draw.line((px, top, px, bottom), fill="#e5e7eb", width=1)
        draw.line((px, bottom, px, bottom + 6), fill="#111827", width=1)
        label = f"{x_val:.0f}" if x_max - x_min >= 5 else f"{x_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((px - (bbox[2] - bbox[0]) / 2, bottom + 10), label, font=tick_font, fill="#111827")

        y_val = y_min + (y_max - y_min) * i / 5
        py = bottom - (bottom - top) * i / 5
        draw.line((left, py, right, py), fill="#e5e7eb", width=1)
        draw.line((left - 6, py, left, py), fill="#111827", width=1)
        label = f"{y_val:.0f}" if y_max - y_min >= 5 else f"{y_val:.1f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 12 - (bbox[2] - bbox[0]), py - (bbox[3] - bbox[1]) / 2), label, font=tick_font, fill="#111827")

    draw_centered_text(draw, ((left + right) / 2, bottom + 55), x_label, label_font)
    draw.text((18, (top + bottom) / 2 - 12), y_label, font=label_font, fill="#111827")

    def to_point(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
        return px, py

    return to_point


def save_regression_plot(df: pd.DataFrame, model: LinearModel, output_path: Path) -> None:
    img = PILImage.new("RGB", (1100, 720), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    to_point = draw_axes(
        draw,
        (105, 80, 1040, 620),
        "Du bao diem cuoi ky tu diem giua ky",
        "Diem giua ky",
        "Diem cuoi ky",
    )

    for x_val, y_val in zip(df["midterm"], df["final"]):
        px, py = to_point(float(x_val), float(y_val))
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#2563ebAA")

    x_grid = np.linspace(0, 10, 120)
    y_grid = clip_scores(model.predict(x_grid))
    points = [to_point(float(x), float(y)) for x, y in zip(x_grid, y_grid)]
    draw.line(points, fill="#dc2626", width=4)

    legend_font = load_plot_font(16)
    draw.rectangle((760, 98, 1030, 160), fill="#ffffffE8", outline="#cbd5e1")
    draw.ellipse((780, 116, 792, 128), fill="#2563ebAA")
    draw.text((802, 111), "Du lieu thuc te", font=legend_font, fill="#111827")
    draw.line((780, 145, 792, 145), fill="#dc2626", width=4)
    draw.text((802, 136), "Duong hoi quy", font=legend_font, fill="#111827")
    img.save(output_path)


def save_residual_plot(test_df: pd.DataFrame, predictions: np.ndarray, output_path: Path) -> None:
    residuals = test_df["final"].to_numpy() - predictions
    y_abs = max(1.0, float(np.max(np.abs(residuals))) + 0.25)
    img = PILImage.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    to_point = draw_axes(
        draw,
        (105, 80, 1040, 550),
        "Phan bo sai so tren tap kiem tra",
        "Diem du bao",
        "Sai so",
        x_min=0,
        x_max=10,
        y_min=-y_abs,
        y_max=y_abs,
    )
    x0, y0 = to_point(0, 0)
    x1, y1 = to_point(10, 0)
    draw.line((x0, y0, x1, y1), fill="#111827", width=2)
    for x_val, y_val in zip(predictions, residuals):
        px, py = to_point(float(x_val), float(y_val))
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="#0f766eCC")
    img.save(output_path)


def save_loss_plot(losses: list[float], output_path: Path) -> None:
    img = PILImage.new("RGB", (1100, 620), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    y_max = max(losses) if losses else 1.0
    y_min = min(losses) if losses else 0.0
    if y_max == y_min:
        y_max += 1.0
    to_point = draw_axes(
        draw,
        (105, 80, 1040, 530),
        "Qua trinh giam loss khi huan luyen computational graph",
        "Buoc ghi nhan",
        "MSE loss",
        x_min=0,
        x_max=max(1, len(losses) - 1),
        y_min=y_min,
        y_max=y_max,
    )
    points = [to_point(i, loss) for i, loss in enumerate(losses)]
    if len(points) > 1:
        draw.line(points, fill="#7c3aed", width=4)
    for px, py in points[:: max(1, len(points) // 35)]:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#7c3aedCC")
    img.save(output_path)


def save_computation_graph(output_path: Path) -> None:
    img = PILImage.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_plot_font(16)
    title_font = load_plot_font(24, bold=True)
    draw_centered_text(draw, (700, 35), "TensorFlow computational graph cho hoi quy tuyen tinh", title_font)
    nodes = {
        "x": (35, 215, 155, 305, "Input\nx"),
        "norm": (220, 205, 390, 315, "Keras\nNormalization"),
        "mu_sigma": (220, 385, 390, 485, "adapt()\nmean, variance"),
        "w": (445, 75, 575, 155, "Dense\nkernel w"),
        "mul": (445, 205, 575, 315, "Tensor op\nw*z"),
        "b": (630, 385, 760, 465, "Dense\nbias b"),
        "add": (630, 205, 795, 315, "Dense output\ny_hat=w*z+b"),
        "yhat": (850, 215, 970, 305, "Du bao\ny_hat"),
        "y": (1005, 385, 1145, 485, "Nhan thuc\nfinal y"),
        "err": (1005, 205, 1145, 315, "Sai so\ne=y_hat-y"),
        "sq": (1195, 205, 1310, 315, "Binh phuong\ne^2"),
        "loss": (1335, 205, 1390, 315, "Mean\nMSE"),
    }
    edges = [
        ("x", "right", "norm", "left"),
        ("mu_sigma", "top", "norm", "bottom"),
        ("norm", "right", "mul", "left"),
        ("w", "bottom", "mul", "top"),
        ("mul", "right", "add", "left"),
        ("b", "top", "add", "bottom"),
        ("add", "right", "yhat", "left"),
        ("yhat", "right", "err", "left"),
        ("y", "top", "err", "bottom"),
        ("err", "right", "sq", "left"),
        ("sq", "right", "loss", "left"),
    ]

    for _, (x0, y0, x1, y1, label) in nodes.items():
        draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill="#f8fafc", outline="#334155", width=2)
        draw_centered_text(
            draw,
            ((x0 + x1) / 2, (y0 + y1) / 2),
            label,
            font,
        )
    def anchor(node_key: str, side: str) -> tuple[float, float]:
        x0, y0, x1, y1, _ = nodes[node_key]
        if side == "left":
            return x0, (y0 + y1) / 2
        if side == "right":
            return x1, (y0 + y1) / 2
        if side == "top":
            return (x0 + x1) / 2, y0
        return (x0 + x1) / 2, y1

    def draw_arrow(start_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
        draw.line((*start_xy, *end_xy), fill="#334155", width=3)
        sx, sy = start_xy
        ex, ey = end_xy
        angle = math.atan2(ey - sy, ex - sx)
        size = 12
        left = (ex - size * math.cos(angle - math.pi / 6), ey - size * math.sin(angle - math.pi / 6))
        right = (ex - size * math.cos(angle + math.pi / 6), ey - size * math.sin(angle + math.pi / 6))
        draw.polygon([end_xy, left, right], fill="#334155")

    for start, start_side, end, end_side in edges:
        draw_arrow(anchor(start, start_side), anchor(end, end_side))
    note_font = load_plot_font(15)
    draw.text(
        (35, 540),
        "Batch Gradient Descent: tinh gradient tren toan bo tap train roi cap nhat tham so mot lan moi epoch",
        font=note_font,
        fill="#111827",
    )
    img.save(output_path)


def run(midterm: float | None = None) -> dict[str, object]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = read_dataset(DATA_PATH)
    train_df, test_df = split_train_test(df)
    model = fit_tensorflow_graph(train_df["midterm"].to_numpy(), train_df["final"].to_numpy())

    test_pred_raw = model.predict(test_df["midterm"].to_numpy())
    test_pred = clip_scores(test_pred_raw)
    metrics = compute_metrics(test_df["final"].to_numpy(), test_pred)

    regression_plot = OUTPUT_DIR / "regression_plot.png"
    residual_plot = OUTPUT_DIR / "residual_plot.png"
    loss_plot = OUTPUT_DIR / "loss_plot.png"
    computation_graph = OUTPUT_DIR / "computation_graph.png"

    save_regression_plot(df, model, regression_plot)
    save_residual_plot(test_df, test_pred, residual_plot)
    save_loss_plot(model.losses, loss_plot)
    save_computation_graph(computation_graph)

    predictions = test_df.copy()
    predictions["predicted_final"] = test_pred
    predictions["error"] = predictions["final"] - predictions["predicted_final"]
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    result: dict[str, object] = {
        "dataset": str(DATA_PATH),
        "n_samples": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "formula": {
            "intercept": model.intercept,
            "slope": model.slope,
            "graph_bias": model.graph_bias,
            "graph_weight": model.graph_weight,
            "feature_mean": model.feature_mean,
            "feature_std": model.feature_std,
            "text": f"final = {model.intercept:.6f} + {model.slope:.6f} * midterm",
        },
        "metrics": metrics,
        "training": {
            "method": "tensorflow_computational_graph_batch_gradient_descent",
            "loss_first": model.losses[0],
            "loss_last": model.losses[-1],
        },
    }
    if midterm is not None:
        result["input_midterm"] = midterm
        result["predicted_final"] = float(clip_scores(np.array([model.predict(np.array([midterm]))[0]]))[0])

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Du bao diem cuoi ky dua tren diem giua ky.")
    parser.add_argument("--midterm", type=float, default=None, help="Diem giua ky can du bao, trong khoang 0-10.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.midterm is not None and not 0 <= args.midterm <= 10:
        raise ValueError("--midterm must be between 0 and 10.")
    result = run(args.midterm)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
