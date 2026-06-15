import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from scipy.cluster.hierarchy import linkage, dendrogram

os.makedirs("charts_gd2", exist_ok=True)

df = pd.read_csv("learnx.csv", nrows=200000)

df = df.drop_duplicates()

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

features = [
    "avg_session_minutes",
    "sessions_per_week",
    "videos_watched",
    "quizzes_taken",
    "completion_rate",
    "courses_enrolled",
    "assignments_submitted",
    "total_spent_usd",
    "ai_recommend_click",
    "ai_recommend_enroll"
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []

for k in range(2, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(2, 8), inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("So cum")
plt.ylabel("Inertia")
plt.savefig("charts_gd2/elbow_method.png")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_mean = df.groupby("cluster")[features].mean()
print(cluster_mean)

cluster_names = {}

for cluster in cluster_mean.index:
    row = cluster_mean.loc[cluster]

    if row["avg_session_minutes"] >= cluster_mean["avg_session_minutes"].mean() and row["completion_rate"] >= cluster_mean["completion_rate"].mean():
        cluster_names[cluster] = "Power Learners"
    elif row["quizzes_taken"] >= cluster_mean["quizzes_taken"].mean():
        cluster_names[cluster] = "Certificate Hunters"
    elif row["sessions_per_week"] <= cluster_mean["sessions_per_week"].mean() and row["avg_session_minutes"] <= cluster_mean["avg_session_minutes"].mean():
        cluster_names[cluster] = "Passive Users"
    else:
        cluster_names[cluster] = "Casual Learners"

df["group_name"] = df["cluster"].map(cluster_names)

print(df["group_name"].value_counts())

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="avg_session_minutes",
    y="completion_rate",
    hue="group_name"
)
plt.title("Thoi gian hoc va completion rate")
plt.savefig("charts_gd2/thoi_gian_hoc_completion_rate.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="future_purchase",
    y="videos_watched"
)
plt.title("So video xem va kha nang mua khoa hoc")
plt.savefig("charts_gd2/video_future_purchase.png")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="ai_recommend_click",
    y="ai_recommend_enroll",
    hue="group_name"
)
plt.title("AI recommendation click va enroll")
plt.savefig("charts_gd2/ai_recommendation.png")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="group_name")
plt.title("So luong nguoi dung theo nhom")
plt.xticks(rotation=20)
plt.savefig("charts_gd2/so_luong_nguoi_dung_theo_nhom.png")
plt.show()

radar_data = cluster_mean[
    [
        "avg_session_minutes",
        "sessions_per_week",
        "videos_watched",
        "quizzes_taken",
        "completion_rate"
    ]
]

radar_scaled = pd.DataFrame(
    scaler.fit_transform(radar_data),
    columns=radar_data.columns,
    index=radar_data.index
)

labels = radar_scaled.columns.tolist()
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for cluster in radar_scaled.index:
    values = radar_scaled.loc[cluster].tolist()
    values += values[:1]
    ax.plot(angles, values, label=cluster_names[cluster])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Star Glyphs hanh vi nguoi dung")
plt.legend()
plt.savefig("charts_gd2/star_glyphs.png")
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(14,4))

for i, cluster in enumerate(cluster_mean.index):
    ax = axes[i]
    row = cluster_mean.loc[cluster]

    face = plt.Circle((0.5, 0.5), 0.35, fill=False)
    ax.add_patch(face)

    eye_size = min(row["completion_rate"] / 100, 0.2)
    ax.plot(0.38, 0.6, "o", markersize=5 + eye_size * 20)
    ax.plot(0.62, 0.6, "o", markersize=5 + eye_size * 20)

    mouth = row["avg_session_minutes"] / cluster_mean["avg_session_minutes"].max()

    if mouth > 0.5:
        ax.plot([0.35, 0.5, 0.65], [0.35, 0.25, 0.35])
    else:
        ax.plot([0.35, 0.5, 0.65], [0.3, 0.38, 0.3])

    ax.set_title(cluster_names[cluster])
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis("off")

plt.savefig("charts_gd2/chernoff_faces.png")
plt.show()

treemap_data = df["group_name"].value_counts()

plt.figure(figsize=(8,6))
plt.pie(
    treemap_data.values,
    labels=treemap_data.index,
    autopct="%1.1f%%"
)
plt.title("Treemap dang don gian ty le cac nhom nguoi dung")
plt.savefig("charts_gd2/treemap_nhom_nguoi_dung.png")
plt.show()

sample = X_scaled[:1000]
Z = linkage(sample, method="ward")

plt.figure(figsize=(12,6))
dendrogram(Z, truncate_mode="lastp", p=30)
plt.title("Dendrogram cau truc nhom hanh vi")
plt.savefig("charts_gd2/dendrogram.png")
plt.show()

model_features = [
    "age",
    "signup_days_ago",
    "sessions_per_week",
    "avg_session_minutes",
    "videos_watched",
    "quizzes_taken",
    "forum_posts",
    "completion_rate",
    "courses_enrolled",
    "assignments_submitted",
    "total_spent_usd",
    "discount_used",
    "ai_recommend_click",
    "ai_recommend_enroll"
]

X_model = df[model_features]
y_purchase = df["future_purchase"]
y_churn = df["churn_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_purchase,
    test_size=0.2,
    random_state=42
)

tree_purchase = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_purchase.fit(X_train, y_train)

pred_purchase = tree_purchase.predict(X_test)

print("DU DOAN FUTURE PURCHASE")
print("Accuracy:", accuracy_score(y_test, pred_purchase))
print(classification_report(y_test, pred_purchase))

plt.figure(figsize=(18,10))
plot_tree(
    tree_purchase,
    feature_names=model_features,
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree du doan future_purchase")
plt.savefig("charts_gd2/tree_future_purchase.png")
plt.show()

importance = pd.DataFrame({
    "feature": model_features,
    "importance": tree_purchase.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importance, x="importance", y="feature")
plt.title("Feature Importance future_purchase")
plt.savefig("charts_gd2/feature_importance_future_purchase.png")
plt.show()

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_model,
    y_churn,
    test_size=0.2,
    random_state=42
)

tree_churn = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_churn.fit(X_train2, y_train2)

pred_churn = tree_churn.predict(X_test2)

print("DU DOAN CHURN RISK")
print("Accuracy:", accuracy_score(y_test2, pred_churn))
print(classification_report(y_test2, pred_churn))

plt.figure(figsize=(18,10))
plot_tree(
    tree_churn,
    feature_names=model_features,
    class_names=["No", "Yes"],
    filled=True
)
plt.title("Decision Tree du doan churn_risk")
plt.savefig("charts_gd2/tree_churn_risk.png")
plt.show()

importance_churn = pd.DataFrame({
    "feature": model_features,
    "importance": tree_churn.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=importance_churn, x="importance", y="feature")
plt.title("Feature Importance churn_risk")
plt.savefig("charts_gd2/feature_importance_churn_risk.png")
plt.show()

try:
    import shap

    shap_sample = X_model.sample(1000, random_state=42)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_model, y_purchase)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(shap_sample)

    shap.summary_plot(shap_values, shap_sample, show=False)
    plt.savefig("charts_gd2/shap_future_purchase.png")
    plt.show()

except:
    print("Chua cai SHAP, neu can chay lenh: pip install shap")

report = open("bao_cao_giai_doan_2_va_hoan_thien.txt", "w", encoding="utf-8")

report.write("BAO CAO GIAI DOAN 2 VA GIAI DOAN HOAN THIEN\n\n")

report.write("1. So nhom nguoi dung chinh\n")
report.write("Sau khi ap dung K-Means, LearnX co 4 nhom nguoi dung chinh.\n\n")

report.write("2. Dac diem tung nhom\n")

for cluster, name in cluster_names.items():
    report.write(f"{name}:\n")
    report.write(str(cluster_mean.loc[cluster]))
    report.write("\n\n")

report.write("3. Goi y cho Product Team\n")
report.write("Power Learners: nen goi y khoa hoc nang cao, chung chi va goi premium.\n")
report.write("Casual Learners: nen gui thong bao nhac hoc va noi dung ngan gon de tang tan suat hoc.\n")
report.write("Certificate Hunters: nen tap trung vao quiz, bai tap va chung chi hoan thanh.\n")
report.write("Passive Users: nen gui email kich hoat lai, voucher, ho tro onboarding.\n\n")

report.write("4. Dashboard\n")
report.write("Da tao cac bieu do phan phoi hanh vi, ket qua phan cum, xu huong hoc tap va bo loc thong qua file dashboard_streamlit.py.\n\n")

report.write("5. Mo hinh du doan\n")
report.write("Da xay dung Decision Tree de du doan future_purchase va churn_risk.\n")
report.write("Da truc quan hoa Feature Importance va Tree Visualization.\n")
report.write("Neu cai them SHAP, chuong trinh se tao bieu do SHAP de giai thich mo hinh.\n")

report.close()

dashboard_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("learnx_clustered.csv")

st.title("LearnX User Behavior Dashboard")

group = st.multiselect(
    "Chon nhom nguoi dung",
    df["group_name"].unique(),
    default=df["group_name"].unique()
)

data = df[df["group_name"].isin(group)]

st.subheader("Phan phoi hanh vi nguoi dung")
fig1 = px.histogram(data, x="avg_session_minutes", color="group_name")
st.plotly_chart(fig1)

st.subheader("Ket qua phan cum")
fig2 = px.scatter(
    data,
    x="avg_session_minutes",
    y="completion_rate",
    color="group_name"
)
st.plotly_chart(fig2)

st.subheader("Xu huong hoc tap")
fig3 = px.scatter(
    data,
    x="sessions_per_week",
    y="videos_watched",
    color="group_name"
)
st.plotly_chart(fig3)

st.subheader("Bo loc tuong tac")
st.dataframe(data)
'''

df.to_csv("learnx_clustered.csv", index=False)

with open("dashboard_streamlit.py", "w", encoding="utf-8") as f:
    f.write(dashboard_code)

print("HOAN THANH GIAI DOAN 2 VA GIAI DOAN HOAN THIEN")
print("Bieu do nam trong thu muc charts_gd2")
print("Bao cao nam trong file bao_cao_giai_doan_2_va_hoan_thien.txt")
print("Dashboard nam trong file dashboard_streamlit.py")
print("Du lieu phan cum nam trong file learnx_clustered.csv")