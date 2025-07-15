# breast-cancer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
df = pd.read_csv("breast-cancer.csv")

# 2. Inspect the data
print("🔍 First 5 rows:\n", df.head())
print("\n📐 Shape:", df.shape)
print("\n🧾 Column Names:\n", df.columns)
print("\n🔍 Info:\n")
df.info()

# 3. Handle missing values (replace "?" with NaN)
df.replace("?", np.nan, inplace=True)

# 4. Convert numerical columns from object to float
for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = df[col].astype(float)
        except:
            pass

# 5. Drop rows with missing values
print("\n❌ Missing values before dropping:\n", df.isnull().sum())
df.dropna(inplace=True)
print("\n✅ Missing values after dropping:\n", df.isnull().sum())

# 6. Remove duplicates
df.drop_duplicates(inplace=True)

# 7. Rename target column and map class labels (2: benign → 0, 4: malignant → 1)
if 'Class' in df.columns:
    df['Class'] = df['Class'].map({2: 0, 4: 1})

# 8. Identify categorical vs numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\n📊 Categorical Columns:", categorical_cols)
print("📈 Numerical Columns:", numerical_cols)

# 9. Label Encode categorical columns (if any remain)
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 10. Correlation Matrix
plt.figure(figsize=(1
