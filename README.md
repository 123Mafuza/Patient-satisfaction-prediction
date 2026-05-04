# 🏥 Patient Satisfaction Predictor

A machine learning project that predicts whether hospital patients are **Satisfied** or **Not Satisfied** based on age and service type, using a Random Forest classifier.

-----

## 📊 What It Does

- Loads patient data from a CSV file
- Engineers a binary satisfaction label (score ≥ 70 = Satisfied)
- Trains a **Random Forest** classifier on age and service type
- Outputs a **classification report** saved to `outputs/`
- Generates a **bar chart** of satisfaction counts saved to `outputs/`

-----

## 📁 Project Structure

```
├── data/
│   └── patients.csv           # Input dataset (required)
├── outputs/
│   ├── classification_report.txt   # Model performance metrics
│   └── satisfaction_chart.png      # Bar chart visualization
├── patient_satisfaction.py    # Main script
├── requirements.txt           # Python dependencies
└── README.md
```

-----

## 🗂️ Dataset Format

Your `data/patients.csv` must include these columns:

|Column        |Type|Description                                   |
|--------------|----|----------------------------------------------|
|`age`         |int |Patient age                                   |
|`service`     |str |Type of service received (e.g. Cardiology, ER)|
|`satisfaction`|int |Satisfaction score (0–100)                    |

-----

## ⚙️ Setup & Usage

**1. Clone the repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your dataset**

Place your `patients.csv` file inside the `data/` folder.

**4. Run the script**

```bash
python patient_satisfaction.py
```

Results will be saved to the `outputs/` folder.

-----

## 📦 Requirements

```
pandas
scikit-learn
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

-----

## 🤖 GitHub Actions (Optional)

To run this automatically on every push, add `.github/workflows/run.yml`:

```yaml
name: Run Patient Satisfaction Model

on: [push]

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python patient_satisfaction.py
```

-----

## 📈 Sample Output

**Classification Report** (`outputs/classification_report.txt`):

```
              precision    recall  f1-score   support

Not Satisfied     0.85      0.82      0.83       120
    Satisfied     0.88      0.90      0.89       180

     accuracy                         0.87       300
```

**Bar Chart** (`outputs/satisfaction_chart.png`):

A bar chart comparing the count of Satisfied vs Not Satisfied patients in the dataset.

-----

## 🛠️ Model Details

|Parameter   |Value                            |
|------------|---------------------------------|
|Algorithm   |Random Forest Classifier         |
|Test split  |30%                              |
|Random state|42                               |
|Features    |Age, Service (one-hot encoded)   |
|Target      |Satisfied (1) / Not Satisfied (0)|

-----

## 📝 License

This project is open source and available under the [MIT License](LICENSE).
