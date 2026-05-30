import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("charts", exist_ok=True)

df = pd.read_csv("learnx.csv")

print("THONG TIN DU LIEU")
print("So dong, so cot:", df.shape)
print("\nCac cot trong du lieu:")
print(df.columns.tolist())
print("\n5 dong dau:")
print(df.head())

print("\nKIEM TRA GIA TRI THIEU")
print(df.isnull().sum())

print("\nKIEM TRA DU LIEU TRUNG LAP")
print("So dong trung lap:", df.duplicated().sum())

df = df.drop_duplicates()

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

print("\nMO TA CAC COT SO")
print(df[numeric_cols].describe())

def find_col(names):
    for name in names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None

study_col = "avg_session_minutes"
visit_col = "sessions_per_week"
completion_col = "completion_rate"
course_col = "courses_enrolled"
spend_col = "total_spent_usd"

print("\nCAC COT DUOC CHON DE PHAN TICH")
print("Thoi gian hoc:", study_col)
print("So lan truy cap:", visit_col)
print("Muc do hoan thanh:", completion_col)
print("Thoi gian: Khong co cot ngay thang")
print("So khoa dang ky:", course_col)
print("Chi tieu:", spend_col)

if study_col:
    plt.figure(figsize=(8,5))
    sns.histplot(df[study_col], bins=20, kde=True)
    plt.title("Phan phoi thoi gian hoc")
    plt.xlabel(study_col)
    plt.ylabel("So luong nguoi dung")
    plt.savefig("charts/phan_phoi_thoi_gian_hoc.png")
    plt.show()

if visit_col:
    plt.figure(figsize=(8,5))
    sns.histplot(df[visit_col], bins=20, kde=True)
    plt.title("So lan truy cap moi tuan")
    plt.xlabel(visit_col)
    plt.ylabel("So luong nguoi dung")
    plt.savefig("charts/so_lan_truy_cap.png")
    plt.show()

if completion_col:
    plt.figure(figsize=(8,5))
    sns.histplot(df[completion_col], bins=20, kde=True)
    plt.title("Muc do hoan thanh khoa hoc")
    plt.xlabel(completion_col)
    plt.ylabel("So luong nguoi dung")
    plt.savefig("charts/muc_do_hoan_thanh.png")
    plt.show()

selected_cols = []

for col in [study_col, visit_col, completion_col, course_col, spend_col]:
    if col and col not in selected_cols:
        selected_cols.append(col)

if len(selected_cols) > 0:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df[selected_cols])
    plt.title("Boxplot phat hien outliers")
    plt.xticks(rotation=30)
    plt.savefig("charts/boxplot_outliers.png")
    plt.show()

outlier_result = pd.DataFrame()

for col in selected_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    temp = df[(df[col] < lower) | (df[col] > upper)].copy()
    temp["outlier_column"] = col
    temp["outlier_value"] = temp[col]

    outlier_result = pd.concat([outlier_result, temp], ignore_index=True)

print("\nSO LUONG OUTLIERS PHAT HIEN:", len(outlier_result))

if len(outlier_result) > 0:
    outlier_result.to_csv("outliers_learnx.csv", index=False)
    print("Da luu file outliers_learnx.csv")

if study_col:
    high_study = df[df[study_col] > df[study_col].quantile(0.95)]
    print("\nNguoi dung hoc cuc ky nhieu:", len(high_study))

if course_col and study_col:
    many_course_no_study = df[
        (df[course_col] > df[course_col].quantile(0.75)) &
        (df[study_col] <= df[study_col].quantile(0.25))
    ]
    print("Nguoi dung dang ky nhieu khoa nhung hoc it:", len(many_course_no_study))

if spend_col:
    abnormal_spend = df[df[spend_col] > df[spend_col].quantile(0.95)]
    print("Nguoi dung chi tieu bat thuong:", len(abnormal_spend))

with open("bao_cao_giai_doan_1.txt", "w", encoding="utf-8") as f:
    f.write("BAO CAO GIAI DOAN 1 - KHAM PHA DU LIEU LEARNX\n\n")
    f.write("1. Gioi thieu du lieu\n")
    f.write(f"So luong ban ghi: {df.shape[0]}\n")
    f.write(f"So luong thuoc tinh: {df.shape[1]}\n")
    f.write(f"Cac thuoc tinh: {df.columns.tolist()}\n\n")

    f.write("2. Lam sach du lieu\n")
    f.write("Da kiem tra missing values, du lieu trung lap va thay the gia tri thieu.\n")
    f.write("Da xoa cac dong trung lap trong du lieu.\n\n")

    f.write("3. Phan tich bang bieu do\n")
    f.write("Da ve bieu do phan phoi thoi gian hoc, so lan truy cap, muc do hoan thanh khoa hoc.\n")
    f.write("Da ve boxplot de phat hien outliers.\n")
    f.write("Da ve xu huong hoc theo thoi gian neu du lieu co cot ngay thang.\n\n")

    f.write("4. Phat hien hanh vi bat thuong\n")
    f.write(f"So luong outliers phat hien: {len(outlier_result)}\n")

    if study_col:
        f.write(f"Nguoi dung hoc cuc ky nhieu: {len(high_study)}\n")

    if course_col and study_col:
        f.write(f"Nguoi dung dang ky nhieu khoa nhung hoc it: {len(many_course_no_study)}\n")

    if spend_col:
        f.write(f"Nguoi dung chi tieu bat thuong: {len(abnormal_spend)}\n")

    f.write("\n5. Insight cho doi san pham\n")
    f.write("Can quan tam den nhom nguoi dung hoc rat it sau khi dang ky.\n")
    f.write("Can co chien luoc khuyen khich nguoi dung hoan thanh khoa hoc.\n")
    f.write("Nhung nguoi dung hoc nhieu co the duoc goi y khoa hoc nang cao.\n")
    f.write("Nhung nguoi dung chi tieu bat thuong can duoc phan tich rieng de toi uu doanh thu.\n")

print("\nHOAN THANH GIAI DOAN 1")
print("Cac bieu do nam trong thu muc charts")
print("Bao cao nam trong file bao_cao_giai_doan_1.txt")