# Store Sales Time Series Forecasting

An end-to-end **data engineering and machine learning pipeline** for the Kaggle **Store Sales - Time Series Forecasting** dataset from Corporación Favorita.

This project combines **ETL, SQLite data warehousing, time-series feature engineering, statistical forecasting, machine learning, model evaluation, and automated reporting** into one reproducible Python pipeline.

---

## 📌 Overview

The goal of this project is to forecast retail sales using historical sales data together with store information, promotions, holidays, transactions, and oil prices.

Instead of relying on a single forecasting model, the pipeline evaluates multiple approaches on the **same validation window** and compares their performance using **RMSLE (Root Mean Squared Logarithmic Error)**.

### Pipeline Architecture

```text
                    Raw Kaggle CSVs
                           │
                           ▼
                    ┌─────────────┐
                    │   Extract   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Transform  │
                    │ Clean / Type│
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   SQLite    │
                    │  Database   │
                    └──────┬──────┘
                           │
                           ▼
                 ┌─────────────────────┐
                 │ Feature Engineering │
                 │ Calendar / Holiday  │
                 │ Oil / Lag / Rolling │
                 └──────────┬──────────┘
                            │
                            ▼
                    ┌─────────────┐
                    │   Modeling  │
                    └──────┬──────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Baselines      SARIMA       LightGBM
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    ┌─────────────┐
                    │ Evaluation  │
                    │    RMSLE    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Reporting  │
                    │ Excel / CSV │
                    └─────────────┘
```

---

# 🎯 Objectives

* Build a reproducible end-to-end sales forecasting pipeline.
* Implement an ETL workflow for multiple related datasets.
* Clean and standardize raw data using Python.
* Store transformed datasets in a **SQLite relational database**.
* Create calendar, holiday, oil-price, lag, and rolling features.
* Train both statistical and machine learning forecasting models.
* Compare models using a consistent validation period.
* Generate automated Excel and CSV reports.
* Produce a reusable command-line pipeline instead of relying on notebooks.

---

# 📊 Dataset

The project uses the **Store Sales - Time Series Forecasting** dataset from Kaggle.

The dataset contains historical sales information for stores and product families in Ecuador, along with additional contextual information.

### Input datasets

| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
| `train.csv`           | Historical store and product-family sales             |
| `test.csv`            | Dates for which sales predictions are required        |
| `stores.csv`          | Store metadata such as city, state, type, and cluster |
| `oil.csv`             | Daily oil prices                                      |
| `holidays_events.csv` | Holidays and special events                           |
| `transactions.csv`    | Daily store transaction counts                        |

The raw dataset is **not included in this repository**. Download the dataset from Kaggle and place the required CSV files inside the `data/` directory.

---

# 🗂️ Project Structure

```text
store-sales-time-series-forecasting/
│
├── Sales.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   └── transactions.csv
│
└── outputs/
    ├── store_sales_etl.db
    ├── store_sales_report.xlsx
    └── model_comparison.csv
```

---

# 🔄 ETL Pipeline

## 1. Extract

The `extract()` function validates and loads the six required Kaggle CSV files.

```python
def extract(data_dir: Path) -> dict[str, pd.DataFrame]:
```

Before loading the data, the pipeline checks that every required file exists.

The datasets are then loaded into Pandas DataFrames.

Example:

```bash
python Sales.py --data-dir "./data"
```

---

## 2. Transform

The `transform()` stage performs dataset-specific cleaning and type conversion.

### General cleaning

* Standardizes column names.
* Removes duplicate records.
* Converts dates to Pandas datetime objects.
* Converts numeric columns to appropriate numeric types.

### Oil price processing

The oil-price dataset is reindexed to a complete daily calendar.

Missing oil prices are filled using interpolation followed by forward/backward filling.

### Store data

Store categorical fields such as:

* City
* State
* Store type

are stripped of unnecessary whitespace.

### Holiday data

Holiday fields are cleaned and converted to appropriate types, including the `transferred` flag.

---

# 🗄️ SQLite Data Layer

After transformation, the cleaned datasets are loaded into a SQLite database.

```text
outputs/store_sales_etl.db
```

The database contains:

```text
train
test
stores
oil
holidays_events
transactions
```

SQLite acts as the **single source of truth for downstream SQL reporting**.

The project therefore demonstrates both:

* DataFrame-based data processing
* Relational database storage and SQL analytics

---

# ⚙️ Feature Engineering

The pipeline creates several groups of predictive features.

## Calendar Features

The following features are extracted from the date:

* `year`
* `month`
* `day`
* `dayofweek`
* `weekofyear`
* `is_weekend`
* `is_month_start`
* `is_month_end`

## Holiday Features

A binary `is_holiday` feature identifies relevant holiday dates while excluding transferred holidays and work days.

## Oil Price

Daily oil prices are mapped to the corresponding sales dates.

## Store Information

Store-level information is merged into the sales data, including:

* Store number
* City
* State
* Store type
* Cluster

## Lag Features

Historical sales are used to create:

```text
sales_lag_7
sales_lag_14
sales_lag_28
```

These features capture weekly and longer-term sales patterns.

## Rolling Features

Rolling historical averages are calculated using:

```text
sales_roll_mean_7
sales_roll_mean_28
```

The rolling features are shifted by one period to avoid using the current target value and introducing data leakage.

## Categorical Encoding

The following categorical variables are label encoded:

```text
family
type
city
state
```

The encoders are fitted using values from both training and test datasets to maintain consistent category mappings.

## Scaling

Oil prices are standardized using `StandardScaler`.

---

# 🤖 Forecasting Models

The project evaluates five forecasting approaches.

## 1. Naive Lag-7 Baseline

The prediction is based on sales from seven days earlier.

This provides a simple seasonal baseline.

---

## 2. 28-Day Moving Average

The prediction uses the historical 28-day rolling average.

This provides a smoother baseline and captures broader local trends.

---

## 3. LightGBM — Store × Family

A LightGBM regression model predicts sales at the:

```text
Store × Product Family
```

level.

The model uses the engineered temporal, store, promotion, holiday, oil-price, lag, rolling, and encoded categorical features.

The target is transformed using:

```python
np.log1p(sales)
```

and converted back using:

```python
np.expm1(predictions)
```

This helps the model handle the highly skewed sales distribution.

### Model configuration

```text
learning_rate    = 0.05
num_leaves       = 63
feature_fraction = 0.8
bagging_fraction = 0.8
bagging_freq     = 5
```

Early stopping is used with a patience of 50 rounds.

---

## 4. SARIMA — National Daily Total

SARIMA is trained on the **aggregated national daily sales total**.

Configuration:

```text
Order:
(1, 1, 1)

Seasonal order:
(1, 1, 1, 7)
```

The seasonal period of 7 captures weekly patterns.

---

## 5. LightGBM — National Daily Total

The store × family LightGBM predictions are aggregated by date:

```text
Store × Family predictions
            ↓
      Group by date
            ↓
National daily sales
```

This allows the LightGBM model to be compared fairly against SARIMA, which directly forecasts the national daily total.

---

# 📏 Evaluation Strategy

The evaluation uses **RMSLE**:

```text
RMSLE = sqrt(mean((log(1 + prediction) - log(1 + actual))²))
```

Lower RMSLE indicates better performance.

### Validation Window

The final run used:

```text
Validation period:
2017-07-17 → 2017-08-15

Validation observations:
53,460
```

All row-level models are evaluated on the same validation period.

The LightGBM predictions are additionally aggregated to the national daily level before comparison with SARIMA.

This avoids comparing predictions made at different target levels.

---

# 📈 Results

### Model Comparison

| Model                 | Forecast Level       |        RMSLE |
| --------------------- | -------------------- | -----------: |
| Naive (Lag-7)         | Store × Family       |     0.544817 |
| 28-Day Moving Average | Store × Family       |     0.458082 |
| LightGBM              | Store × Family       | **0.385800** |
| SARIMA                | National Daily Total |     0.108455 |
| LightGBM (Aggregated) | National Daily Total | **0.063080** |

### 🏆 Best Result

The best evaluated model was:

**LightGBM — Aggregated National Daily Total**

```text
RMSLE = 0.063080
```

At the store × family level, LightGBM also substantially outperformed both baseline approaches:

```text
Naive Lag-7       → 0.544817
28-Day Average    → 0.458082
LightGBM          → 0.385800
```

---

# 📊 Automated Reporting

The pipeline automatically generates reports in:

```text
outputs/
```

## Excel Report

```text
store_sales_report.xlsx
```

The workbook contains SQL-generated analysis including:

* Overall sales statistics
* Sales by store
* Sales by product family
* Monthly sales
* Sales by promotion status

## Model Comparison

```text
model_comparison.csv
```

Contains the RMSLE results for all evaluated models.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd store-sales-time-series-forecasting
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Setup

Download the Kaggle Store Sales dataset and place the following files in:

```text
data/
```

Required files:

```text
data/
├── train.csv
├── test.csv
├── stores.csv
├── oil.csv
├── holidays_events.csv
└── transactions.csv
```

---

# ▶️ Running the Pipeline

Run the complete pipeline from the project root:

```bash
python app.py --data-dir "./data"
```

The default validation window is 30 days.

You can specify a different validation period:

```bash
python app.py --data-dir "./data" --validation-days 30
```

You can also specify the output directory:

```bash
python Sales.py \
    --data-dir "./data" \
    --output-dir "./outputs" \
    --validation-days 30
```

### Windows PowerShell

```powershell
python app.py --data-dir ".\data" --output-dir ".\outputs" --validation-days 30
```

---

# 🧪 Reproducibility

The entire workflow is executed through a single command-line Python script.

This makes the project reproducible without requiring separate notebooks for:

* Data extraction
* Data cleaning
* Database loading
* Feature engineering
* Model training
* Evaluation
* Reporting

The same pipeline can be rerun with different validation periods or input data directories.

---

# ⚠️ Known Warning

During SARIMA forecasting, `statsmodels` may produce warnings similar to:

```text
ValueWarning:
A date index has been provided, but it has no associated frequency information
```

The warning does not prevent the pipeline from completing.

A future improvement would be to explicitly assign a frequency to the daily time-series index before fitting SARIMA.

---

# 🔮 Future Improvements

Potential extensions include:

* Hyperparameter optimization for LightGBM
* Rolling time-series cross-validation
* Additional lag and rolling-window features
* More advanced holiday and promotion features
* Improved handling of missing values
* XGBoost and CatBoost comparisons
* Advanced forecasting models such as Temporal Fusion Transformers
* Model experiment tracking
* Model serialization
* Forecasting API using FastAPI
* Interactive dashboard using Streamlit
* Automated CI/CD pipeline
* Docker containerization
* Automated model retraining

---

# 🛠️ Tech Stack

| Category             | Technologies           |
| -------------------- | ---------------------- |
| Language             | Python                 |
| Data Processing      | Pandas, NumPy          |
| Database             | SQLite                 |
| Machine Learning     | LightGBM, Scikit-learn |
| Statistical Modeling | Statsmodels, SARIMA    |
| Data Analysis        | Pandas, SQL            |
| Reporting            | Excel, CSV             |
| Visualization        | Matplotlib, Seaborn    |

---

# 📚 Skills Demonstrated

This project demonstrates practical experience in:

* ETL pipeline development
* Data cleaning and preprocessing
* SQL and relational databases
* SQLite database design
* Time-series analysis
* Feature engineering
* Machine learning
* Statistical forecasting
* Model evaluation
* Data leakage prevention
* Command-line application design
* Automated reporting
* Reproducible ML workflows

---

# 👩‍💻 Author

**Nada Mohamed**

Computer Science Student | AI & Machine Learning

Interested in:

* Artificial Intelligence
* Machine Learning
* Data Science
* Data Engineering
* Healthcare AI

---

# 📄 License

This project is intended for educational and portfolio purposes.

The underlying dataset is provided by Kaggle and remains subject to its original terms of use.
