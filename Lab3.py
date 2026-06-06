import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================
# 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU
# =========================

df = pd.read_csv("data.csv.txt")

print("5 dòng đầu:")
print(df.head())

print("\nThông tin dữ liệu:")
print(df.info())

# Điền dữ liệu thiếu
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Mã hóa dữ liệu chữ thành số
df["Sex_code"] = LabelEncoder().fit_transform(df["Sex"])
df["Embarked_code"] = LabelEncoder().fit_transform(df["Embarked"])

# Chọn các cột số để phân tích
numeric_cols = [
    "Survived", "Pclass", "Age", "SibSp",
    "Parch", "Fare", "Sex_code", "Embarked_code"
]

df_num = df[numeric_cols]

# =========================
# LAB 3 - BÀI 1
# Scatter Plot
# =========================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Age",
    y="Fare",
    hue="Survived",
    palette="Set1"
)
plt.title("LAB 3 - Bài 1: Scatter Plot Age và Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

print("""
Nhận xét Bài 1:
Biểu đồ Scatter Plot thể hiện mối quan hệ giữa tuổi và giá vé.
Màu sắc phân biệt hành khách sống sót và không sống sót.
Có thể thấy giá vé cao thường xuất hiện nhiều hơn ở nhóm sống sót.
""")

# =========================
# LAB 3 - BÀI 2
# Scatter Matrix / Pairplot
# =========================

pair_cols = ["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch"]

sns.pairplot(
    df[pair_cols],
    hue="Survived",
    diag_kind="kde",
    palette="Set2"
)
plt.suptitle("LAB 3 - Bài 2: Scatter Matrix Titanic", y=1.02)
plt.show()

print("""
Nhận xét Bài 2:
Pairplot giúp quan sát mối quan hệ giữa nhiều thuộc tính.
Các biến Fare, Pclass và Age có ảnh hưởng đến khả năng sống sót.
Fare cao và Pclass nhỏ thường liên quan đến tỉ lệ sống sót cao hơn.
""")

# =========================
# LAB 3 - BÀI 3
# Parallel Coordinates
# =========================

features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_code"]

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[features]),
    columns=features
)

df_scaled["Survived"] = df["Survived"].astype(str)

plt.figure(figsize=(12, 6))
parallel_coordinates(
    df_scaled,
    "Survived",
    colormap=plt.cm.Set1,
    linewidth=1,
    alpha=0.5
)
plt.title("LAB 3 - Bài 3: Parallel Coordinates Plot")
plt.xlabel("Features")
plt.ylabel("Normalized Value")
plt.show()

print("""
Nhận xét Bài 3:
Biểu đồ Parallel Coordinates cho thấy sự khác biệt giữa nhóm sống sót và không sống sót.
Các thuộc tính như Sex_code, Fare và Pclass thể hiện sự phân biệt khá rõ.
""")

# =========================
# LAB 3 - BÀI 4
# PCA và t-SNE
# =========================

X = df[features]
y = df["Survived"]

X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="tab10",
    s=20
)
plt.title("LAB 3 - Bài 4: PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Survived")
plt.show()

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="tab10",
    s=20
)
plt.title("LAB 3 - Bài 4: t-SNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(label="Survived")
plt.show()

print("""
Nhận xét Bài 4:
PCA giúp giảm dữ liệu nhiều chiều xuống 2 chiều để dễ quan sát.
t-SNE thường thể hiện cụm dữ liệu rõ hơn PCA.
Trong dữ liệu Titanic, hai nhóm sống sót và không sống sót có sự phân tách nhưng chưa hoàn toàn rõ ràng.
""")