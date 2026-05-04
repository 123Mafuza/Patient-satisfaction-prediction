import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use(‘Agg’)  # Non-interactive backend — required for GitHub Actions / headless environments
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_PATH  = os.path.join(‘data’, ‘patients.csv’)
OUTPUT_DIR = ‘outputs’
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

if not os.path.exists(DATA_PATH):
raise FileNotFoundError(
f”Dataset not found at ‘{DATA_PATH}’. “
“Make sure ‘patients.csv’ is inside a ‘data/’ folder in the repo root.”
)

data = pd.read_csv(DATA_PATH)

# ── Feature engineering ────────────────────────────────────────────────────────

data[‘satisfied’] = (data[‘satisfaction’] >= 70).astype(int)
data[‘label’]     = data[‘satisfied’].map({1: ‘Satisfied’, 0: ‘Not Satisfied’})

X = pd.get_dummies(data[[‘age’, ‘service’]], columns=[‘service’])
y = data[‘satisfied’]

# ── Train / test split ─────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=42
)

# ── Model ──────────────────────────────────────────────────────────────────────

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Classification report ──────────────────────────────────────────────────────

report = classification_report(y_test, y_pred, target_names=[‘Not Satisfied’, ‘Satisfied’])
print(report)

report_path = os.path.join(OUTPUT_DIR, ‘classification_report.txt’)
with open(report_path, ‘w’) as f:
f.write(report)
print(f”Classification report saved → {report_path}”)

# ── Bar chart (saved, not displayed) ──────────────────────────────────────────

label_counts = data[‘label’].value_counts()

fig, ax = plt.subplots()
ax.bar(label_counts.index, label_counts.values, color=[‘orange’, ‘blue’])
ax.set_xlabel(‘Customer Satisfaction’)
ax.set_ylabel(‘Count’)
ax.set_title(‘Counts of Satisfied vs Not Satisfied Customers’)

chart_path = os.path.join(OUTPUT_DIR, ‘satisfaction_chart.png’)
plt.savefig(chart_path, bbox_inches=‘tight’, dpi=150)
plt.close()
print(f”Chart saved → {chart_path}”)
