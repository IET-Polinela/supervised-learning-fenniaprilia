import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Load Dataset
df = pd.read_csv("train.csv")

# 1. Data Understanding
print("\nStatistik Deskriptif:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Preprocessing (Handling Missing Values & Encoding)
# Pisahkan kolom numerik dan kategorikal
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

# Isi nilai yang hilang pada kolom numerik dengan median
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Isi nilai yang hilang pada kolom kategorikal dengan modus (nilai terbanyak)
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Label Encoding untuk fitur kategorikal
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 3. Handling Outliers (Menggunakan IQR)
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1

df_no_outliers = df[~((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Simpan dataset dengan dan tanpa outlier
df.to_csv("dataset_with_outliers.csv", index=False)
df_no_outliers.to_csv("dataset_without_outliers.csv", index=False)

# Visualisasi Boxplot (Outliers)
plt.figure(figsize=(12,6))
sns.boxplot(data=df[numerical_cols])
plt.xticks(rotation=90)
plt.title("Boxplot Dataset dengan Outliers")
plt.savefig("boxplot_with_outliers.png")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df_no_outliers[numerical_cols])
plt.xticks(rotation=90)
plt.title("Boxplot Dataset Tanpa Outliers")
plt.savefig("boxplot_without_outliers.png")
plt.show()

# 4. Feature Selection & Scaling untuk dataset dengan dan tanpa outlier
# Dataset dengan outlier
X_with_outliers = df.drop(columns=['SalePrice'])
Y_with_outliers = df['SalePrice']

X_train_with_outliers, X_test_with_outliers, Y_train_with_outliers, Y_test_with_outliers = train_test_split(
    X_with_outliers, Y_with_outliers, test_size=0.2, random_state=42
)

# Dataset tanpa outlier
X_no_outliers = df_no_outliers.drop(columns=['SalePrice'])
Y_no_outliers = df_no_outliers['SalePrice']

X_train_no_outliers, X_test_no_outliers, Y_train_no_outliers, Y_test_no_outliers = train_test_split(
    X_no_outliers, Y_no_outliers, test_size=0.2, random_state=42
)

# Scaling dengan StandardScaler
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train_no_outliers)
X_test_std = std_scaler.transform(X_test_no_outliers)

# Scaling dengan MinMaxScaler
minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train_no_outliers)
X_test_minmax = minmax_scaler.transform(X_test_no_outliers)

# Visualisasi perbandingan distribusi sebelum dan sesudah scaling
# Pilih beberapa fitur numerik untuk visualisasi (misalnya 4 fitur pertama)
selected_features = numerical_cols[:4]
fig, axes = plt.subplots(4, 3, figsize=(20, 16))
fig.suptitle("Perbandingan Distribusi Sebelum dan Sesudah Scaling", fontsize=16)

for i, feature in enumerate(selected_features):
    feature_idx = list(X_train_no_outliers.columns).index(feature)
    
    # Original data
    axes[i, 0].hist(X_train_no_outliers[feature], bins=30)
    axes[i, 0].set_title(f"{feature} - Original")
    axes[i, 0].set_xlabel("Value")
    axes[i, 0].set_ylabel("Frequency")
    
    # StandardScaler
    axes[i, 1].hist(X_train_std[:, feature_idx], bins=30)
    axes[i, 1].set_title(f"{feature} - StandardScaler")
    axes[i, 1].set_xlabel("Value")
    
    # MinMaxScaler
    axes[i, 2].hist(X_train_minmax[:, feature_idx], bins=30)
    axes[i, 2].set_title(f"{feature} - MinMaxScaler")
    axes[i, 2].set_xlabel("Value")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("scaling_comparison.png")
plt.show()

# 5. Implementation: Linear Regression pada kedua dataset (dengan dan tanpa outlier)
# Dataset dengan outlier
lr_with_outliers = LinearRegression()
lr_with_outliers.fit(X_train_with_outliers, Y_train_with_outliers)
Y_pred_outliers = lr_with_outliers.predict(X_test_with_outliers)
mse_outliers = mean_squared_error(Y_test_with_outliers, Y_pred_outliers)
r2_outliers = r2_score(Y_test_with_outliers, Y_pred_outliers)

# Dataset tanpa outlier (dengan StandardScaler)
lr_no_outliers = LinearRegression()
lr_no_outliers.fit(X_train_std, Y_train_no_outliers)
Y_pred_no_outliers = lr_no_outliers.predict(X_test_std)
mse_no_outliers = mean_squared_error(Y_test_no_outliers, Y_pred_no_outliers)
r2_no_outliers = r2_score(Y_test_no_outliers, Y_pred_no_outliers)

# Visualisasi scatter plot dan residual plot untuk model dengan outlier
plt.figure(figsize=(15, 5))

# Scatter plot hasil prediksi vs nilai aktual
plt.subplot(1, 3, 1)
plt.scatter(Y_test_with_outliers, Y_pred_outliers, alpha=0.5)
plt.plot([Y_test_with_outliers.min(), Y_test_with_outliers.max()], 
         [Y_test_with_outliers.min(), Y_test_with_outliers.max()], 'r--')
plt.xlabel("Nilai Aktual")
plt.ylabel("Nilai Prediksi")
plt.title("Prediksi vs Aktual (Dengan Outlier)")

# Residual plot
residuals_outliers = Y_test_with_outliers - Y_pred_outliers
plt.subplot(1, 3, 2)
plt.scatter(Y_pred_outliers, residuals_outliers, alpha=0.5)
plt.hlines(y=0, xmin=Y_pred_outliers.min(), xmax=Y_pred_outliers.max(), colors='r', linestyles='--')
plt.xlabel("Nilai Prediksi")
plt.ylabel("Residual")
plt.title("Residual Plot (Dengan Outlier)")

# Distribusi residual
plt.subplot(1, 3, 3)
plt.hist(residuals_outliers, bins=30)
plt.xlabel("Residual")
plt.ylabel("Frekuensi")
plt.title("Distribusi Residual (Dengan Outlier)")

plt.tight_layout()
plt.savefig("linear_regression_with_outliers.png")
plt.show()

# Visualisasi scatter plot dan residual plot untuk model tanpa outlier
plt.figure(figsize=(15, 5))

# Scatter plot hasil prediksi vs nilai aktual
plt.subplot(1, 3, 1)
plt.scatter(Y_test_no_outliers, Y_pred_no_outliers, alpha=0.5)
plt.plot([Y_test_no_outliers.min(), Y_test_no_outliers.max()], 
         [Y_test_no_outliers.min(), Y_test_no_outliers.max()], 'r--')
plt.xlabel("Nilai Aktual")
plt.ylabel("Nilai Prediksi")
plt.title("Prediksi vs Aktual (Tanpa Outlier)")

# Residual plot
residuals_no_outliers = Y_test_no_outliers - Y_pred_no_outliers
plt.subplot(1, 3, 2)
plt.scatter(Y_pred_no_outliers, residuals_no_outliers, alpha=0.5)
plt.hlines(y=0, xmin=Y_pred_no_outliers.min(), xmax=Y_pred_no_outliers.max(), colors='r', linestyles='--')
plt.xlabel("Nilai Prediksi")
plt.ylabel("Residual")
plt.title("Residual Plot (Tanpa Outlier)")

# Distribusi residual
plt.subplot(1, 3, 3)
plt.hist(residuals_no_outliers, bins=30)
plt.xlabel("Residual")
plt.ylabel("Frekuensi")
plt.title("Distribusi Residual (Tanpa Outlier)")

plt.tight_layout()
plt.savefig("linear_regression_without_outliers.png")
plt.show()

# 6 & 7. Implementation: Polynomial Regression dan KNN Regression
# Semua model menggunakan dataset tanpa outlier dengan StandardScaler
models = {
    "Linear Regression (dengan outlier)": [LinearRegression(), X_train_with_outliers, Y_train_with_outliers, X_test_with_outliers, Y_test_with_outliers],
    "Linear Regression (tanpa outlier)": [LinearRegression(), X_train_std, Y_train_no_outliers, X_test_std, Y_test_no_outliers],
    "Polynomial Regression (Degree 2)": [make_pipeline(PolynomialFeatures(2), LinearRegression()), X_train_std, Y_train_no_outliers, X_test_std, Y_test_no_outliers],
    "Polynomial Regression (Degree 3)": [make_pipeline(PolynomialFeatures(3), LinearRegression()), X_train_std, Y_train_no_outliers, X_test_std, Y_test_no_outliers],
    "KNN Regression (K=3)": [KNeighborsRegressor(n_neighbors=3), X_train_std, Y_train_no_outliers, X_test_std, Y_test_no_outliers],
    "KNN Regression (K=5)": [KNeighborsRegressor(n_neighbors=5), X_train_std, Y_train_no_outliers, X_test_std, Y_test_no_outliers],
    "KNN Regression (K=7)": [KNeighborsRegressor(n_neighbors=7), X_train_std, Y_train_no_outliers, X_test_std, Y_test_no_outliers]
}

results = []
plt.figure(figsize=(20, 15))

for i, (name, (model, X_tr, Y_tr, X_te, Y_te)) in enumerate(models.items(), 1):
    model.fit(X_tr, Y_tr)
    Y_pred = model.predict(X_te)
    mse = mean_squared_error(Y_te, Y_pred)
    r2 = r2_score(Y_te, Y_pred)
    results.append([name, mse, r2])

    # Scatter plot hasil prediksi vs nilai aktual
    plt.subplot(3, 3, i)
    plt.scatter(Y_te, Y_pred, alpha=0.5)
    plt.plot([Y_te.min(), Y_te.max()], [Y_te.min(), Y_te.max()], 'r--')
    plt.xlabel("Actual Price", fontsize=10)
    plt.ylabel("Predicted Price", fontsize=10)
    plt.title(f"{name}", fontsize=12)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

plt.tight_layout()
plt.savefig("all_models_predictions.png")
plt.show()

# 8. Tabel Perbandingan Model
df_results = pd.DataFrame(results, columns=["Model", "MSE", "R2 Score"])
df_results.to_csv("model_comparison.csv", index=False)
print("\nHasil Evaluasi Model:")
print(df_results)

# Visualisasi Perbandingan MSE dan R2
fig, ax = plt.subplots(1, 2, figsize=(15, 8))

# Bar plot MSE
ax[0].barh(df_results["Model"], df_results["MSE"], color='skyblue')
ax[0].set_xlabel("MSE", fontsize=12)
ax[0].set_title("Comparison of MSE", fontsize=14)
ax[0].tick_params(axis='y', labelsize=10)

# Bar plot R2 Score
ax[1].barh(df_results["Model"], df_results["R2 Score"], color='lightcoral')
ax[1].set_xlabel("R2 Score", fontsize=12)
ax[1].set_title("Comparison of R2 Score", fontsize=14)
ax[1].tick_params(axis='y', labelsize=10)

plt.tight_layout()
plt.savefig("model_comparison_plots.png")
plt.show()
