# ==================================
# ITA106 - LAB 4
# Dataset: data.csv.txt
# ==================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# =========================
# ĐỌC DỮ LIỆU
# =========================

df = pd.read_csv("data.csv.txt")

# Xử lý dữ liệu thiếu
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Mã hóa dữ liệu dạng chữ
df["Sex_code"] = LabelEncoder().fit_transform(df["Sex"])
df["Embarked_code"] = LabelEncoder().fit_transform(df["Embarked"])

# =========================
# BÀI 1: HEATMAP
# =========================

print("\n===== BÀI 1: HEATMAP =====")

corr_matrix = df[
    ["Survived", "Pclass", "Age",
     "SibSp", "Parch", "Fare",
     "Sex_code", "Embarked_code"]
].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()
print("""
Nhận xét Bài 1:
Heatmap cho thấy mối tương quan giữa các biến số.
Survived có tương quan với Sex_code, Fare và Pclass.
Fare và Pclass cũng có mối liên hệ đáng chú ý.
""")
# =========================
# BÀI 2: PIXEL VISUALIZATION
# =========================

print("\n===== BÀI 2: PIXEL VISUALIZATION =====")

values = df["Fare"].values

values = MinMaxScaler().fit_transform(
    values.reshape(-1, 1)
).flatten()

size = int(np.ceil(np.sqrt(len(values))))

pixel_matrix = np.zeros(size * size)

pixel_matrix[:len(values)] = values

pixel_matrix = pixel_matrix.reshape(size, size)

plt.figure(figsize=(8, 8))

plt.imshow(pixel_matrix, cmap="viridis")

plt.colorbar(label="Fare")

plt.title("Pixel-based Visualization")

plt.show()
print("""
Nhận xét Bài 2:
Mỗi pixel biểu diễn một hành khách theo giá vé Fare.
Màu sáng biểu diễn giá vé cao, màu tối biểu diễn giá vé thấp.
Đa số hành khách có giá vé thấp, chỉ một số ít có giá vé rất cao.
""")

# =========================
# BÀI 3: STAR GLYPH
# =========================

print("\n===== BÀI 3: STAR GLYPH =====")

features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Sex_code"
]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df[features])

def star_plot(values, label):

    num_vars = len(values)

    angles = np.linspace(
        0,
        2 * np.pi,
        num_vars,
        endpoint=False
    )

    values = np.concatenate(
        (values, [values[0]])
    )

    angles = np.concatenate(
        (angles, [angles[0]])
    )

    fig = plt.figure(figsize=(5, 5))

    ax = plt.subplot(
        111,
        polar=True
    )

    ax.plot(
        angles,
        values,
        linewidth=2
    )

    ax.fill(
        angles,
        values,
        alpha=0.3
    )

    ax.set_xticks(angles[:-1])

    ax.set_xticklabels(features)

    ax.set_title(label)

    plt.show()

print("""
Nhận xét Bài 3:
Star Glyph giúp biểu diễn nhiều thuộc tính của một hành khách.
Mỗi hình sao thể hiện một mẫu dữ liệu khác nhau.
Có thể so sánh hình dạng để thấy sự khác biệt giữa các hành khách.
""")

# Hiển thị 3 mẫu đầu tiên
for i in range(3):
    star_plot(
        scaled_data[i],
        f"Passenger {i+1}"
    )

# =========================
# BÀI 4: CHERNOFF FACES
# =========================

print("\n===== BÀI 4: CHERNOFF FACES =====")

face_features = [
    "Pclass",
    "Age",
    "Fare",
    "SibSp"
]

samples = scaler.fit_transform(
    df[face_features].head(9)
)

def draw_face(ax, data):

    face_size = 0.5 + data[0] * 0.5

    eye_size = 0.05 + data[1] * 0.05

    mouth_curve = data[2] - 0.5

    nose_size = 0.05 + data[3] * 0.05

    # Khuôn mặt
    face = plt.Circle(
        (0.5, 0.5),
        face_size * 0.35,
        fill=False,
        linewidth=2
    )

    ax.add_patch(face)

    # Mắt
    left_eye = plt.Circle(
        (0.35, 0.6),
        eye_size,
        color="black"
    )

    right_eye = plt.Circle(
        (0.65, 0.6),
        eye_size,
        color="black"
    )

    ax.add_patch(left_eye)
    ax.add_patch(right_eye)

    # Mũi
    nose = plt.Circle(
        (0.5, 0.5),
        nose_size,
        color="black"
    )

    ax.add_patch(nose)

    # Miệng
    x = np.linspace(
        0.35,
        0.65,
        100
    )

    y = 0.35 + mouth_curve * (x - 0.5) ** 2 * -4

    ax.plot(
        x,
        y,
        linewidth=2
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.axis("off")

fig, axes = plt.subplots(
    3,
    3,
    figsize=(10, 10)
)

for i, ax in enumerate(axes.flat):

    draw_face(
        ax,
        samples[i]
    )

    ax.set_title(
        f"Passenger {i+1}"
    )
plt.suptitle(
    "Chernoff Faces - Titanic Dataset",
    fontsize=16
)
plt.tight_layout()
plt.show()

print("""
Nhận xét Bài 4:
Chernoff Faces giúp biểu diễn nhiều thuộc tính của một hành khách dưới dạng khuôn mặt.
Mỗi khuôn mặt thể hiện một mẫu dữ liệu khác nhau.
Có thể so sánh các khuôn mặt để thấy sự khác biệt giữa các hành khách.
""")