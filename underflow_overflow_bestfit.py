# Configure KAGGLE_USERNAME and KAGGLE_KEY in the environment before running.

from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 2. AUTHENTICATE & DOWNLOAD DATA VIA API
# ==========================================
api = KaggleApi()
api.authenticate()

# Download and unzip 'data.csv' into the current folder
api.dataset_download_files(
    'shrutibhargava94/india-air-quality-data',
    path='.',
    unzip=True
)
print("Dataset downloaded and extracted successfully!")

# Load the extracted CSV
df = pd.read_csv('data.csv', encoding='ISO-8859-1', low_memory=False)
print("Initial Shape:", df.shape)

# ==========================================
# 3. DATA CLEANING & PREPROCESSING
# ==========================================
cols_to_drop = ['stn_code', 'agency', 'location_monitoring_station', 'sampling_date', 'pm2_5', 'spm']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

df['type'] = df['type'].replace({
    'Residential, Rural and other Areas': 'Residential',
    'Residential and others': 'Residential',
    'Industrial Area': 'Industrial',
    'Industrial Areas': 'Industrial',
    'Sensitive Area': 'Sensitive',
    'Sensitive Areas': 'Sensitive'
})
df['type'] = df['type'].fillna('Residential')

# Drop rows where all core pollutants are missing
df.dropna(subset=['so2', 'no2', 'rspm'], how='all', inplace=True)

# Impute missing pollutant values using state-wise median, then global median
for col in ['so2', 'no2', 'rspm']:
    df[col] = df[col].fillna(df.groupby('state')[col].transform('median'))
    df[col] = df[col].fillna(df[col].median())

# ==========================================
# 4. FEATURE ENGINEERING: COMPUTE AQI (CPCB)
# ==========================================
def calc_so2_index(so2):
    if so2 <= 40: return so2 * (50 / 40)
    elif so2 <= 80: return 50 + (so2 - 40) * (50 / 40)
    elif so2 <= 380: return 100 + (so2 - 80) * (100 / 300)
    elif so2 <= 800: return 200 + (so2 - 380) * (100 / 420)
    elif so2 <= 1600: return 300 + (so2 - 800) * (100 / 800)
    else: return 400 + (so2 - 1600) * (100 / 800)

def calc_no2_index(no2):
    if no2 <= 40: return no2 * (50 / 40)
    elif no2 <= 80: return 50 + (no2 - 40) * (50 / 40)
    elif no2 <= 180: return 100 + (no2 - 80) * (100 / 100)
    elif no2 <= 280: return 200 + (no2 - 180) * (100 / 100)
    elif no2 <= 400: return 300 + (no2 - 280) * (100 / 120)
    else: return 400 + (no2 - 400) * (100 / 120)

def calc_rspm_index(rspm):
    if rspm <= 50: return rspm
    elif rspm <= 100: return rspm
    elif rspm <= 250: return 100 + (rspm - 100) * (100 / 150)
    elif rspm <= 350: return 200 + (rspm - 250)
    elif rspm <= 430: return 300 + (rspm - 350) * (100 / 80)
    else: return 400 + (rspm - 430) * (100 / 80)

df['SOi'] = df['so2'].apply(calc_so2_index)
df['NOi'] = df['no2'].apply(calc_no2_index)
df['RSPMi'] = df['rspm'].apply(calc_rspm_index)

df['AQI'] = df[['SOi', 'NOi', 'RSPMi']].max(axis=1)

# ==========================================
# 5. ENCODING & TRAIN-TEST SPLIT
# ==========================================
le_state = LabelEncoder()
le_type = LabelEncoder()

df['state_encoded'] = le_state.fit_transform(df['state'].astype(str))
df['type_encoded'] = le_type.fit_transform(df['type'].astype(str))

X = df[['so2', 'no2', 'rspm', 'state_encoded', 'type_encoded']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 6. MODEL TRAINING & EVALUATION
# ==========================================
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("\n--- Model Performance Comparison ---")
print(f"Linear Regression -> R2: {r2_score(y_test, lr_preds):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.2f}")
print(f"Random Forest     -> R2: {r2_score(y_test, rf_preds):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.2f}")

# ==========================================
# 7. VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
top_states = df.groupby('state')['AQI'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_states.values, y=top_states.index, hue=top_states.index, palette='Reds_r', legend=False)
plt.title('Top 10 States by Average AQI')
plt.xlabel('Mean AQI')

plt.subplot(1, 2, 2)
plt.scatter(y_test[:500], rf_preds[:500], alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Random Forest: Actual vs Predicted AQI')
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')

plt.tight_layout()
plt.show()
