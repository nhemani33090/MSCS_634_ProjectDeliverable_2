# Advanced Data Mining Project — Superstore (Deliverables 1 & 2)

## 📘 Dataset Summary and Justification
The dataset used for this project is a **Superstore Sales Dataset** that contains transactional data across multiple product categories, regions, and customer segments.  
It includes approximately **9,994 rows** and **21 attributes**, such as `Sales`, `Profit`, `Discount`, `Quantity`, `Category`, `Region`, and `Segment`.

**Why this dataset is appropriate:**
- It provides both **numeric** (`Sales`, `Profit`, `Discount`, `Quantity`) and **categorical** (`Category`, `Region`, `Segment`) features — ideal for regression, classification, clustering, and association-rule mining in later deliverables.
- It is **clean, well-structured, and interpretable**, enabling clear business insights and predictive modeling.
- It meets the project requirement of having **≥ 8 attributes** and **≥ 500 records**.

---

## 📎 Dataset Source
This project uses the **Superstore Sales** dataset from **Kaggle**:

- Kaggle dataset: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final  
- File used: `Sample - Superstore.csv` (downloaded from the dataset above)

**Attribution:**  
Vivek. (2019). *Superstore Sales Dataset*. Kaggle. Retrieved October 26, 2025.

---

## 🧩 Deliverable 1 — Data Collection, Cleaning, and Exploration

### 🔹 Data Cleaning Steps
Comprehensive data-cleaning operations were performed to ensure quality and readiness for modeling.

1. **Data Loading & Inspection**
   - Loaded with Pandas and reviewed using `.head()`, `.info()`, and `.describe()` to confirm structure and datatypes.

2. **Handling Missing Values**
   - Missing values were summarized **before** and **after** cleaning using a custom `summarize_missing()` function.
   - Numeric columns were imputed with the **median**, categorical columns with the **mode`** (dataset contained no nulls, but logic was present and documented).

3. **Removing Duplicates**
   - Duplicates were checked and removed using `drop_duplicates()` (result: **0** duplicates).

4. **Correcting Inconsistent Data Types**
   - Parsed `Order Date` and `Ship Date` into datetime format.
   - Converted `Postal Code` to string to preserve leading zeros.

5. **Identifying and Addressing Noisy Data (Outliers)**
   - Outlier counts were displayed **before** clipping for `Sales`, `Profit`, `Discount`, and `Quantity`.
   - Clipped outliers using the **IQR method** to reduce noise while maintaining the overall data distribution.

**Challenge & Resolution:**  
Although the dataset was already clean, irregular discount and extreme profit values were identified. The IQR method was applied to cap outliers, preventing distortion of regression models while retaining natural variability.

### 🔹 Exploratory Data Analysis (EDA)
EDA was performed using Matplotlib to visualize data distributions and relationships.

**Visualizations Included:**
- **Histograms & Boxplots** — to understand distributions and detect outliers.
- **Correlation Heatmap** — to explore relationships among `Sales`, `Profit`, `Quantity`, and `Discount`.
- **Grouped Summaries** — mean profit by `Category`, `Sub-Category`, `Region`, and `Segment`.

**Key Insights:**
- Higher discounts generally **reduce profit margins**.
- The **Technology** category and **West region** deliver the **highest average profits**.
- **Furniture sub-categories** such as *Tables* and *Bookcases* show **low profitability**.
- `Profit` correlates **positively with Sales** and **negatively with Discount**, informing future modeling features.

The cleaned dataset is exported as: `data/cleaned_superstore.csv`.

---

## 📈 Deliverable 2 — Regression Modeling and Performance Evaluation

### 🎯 Objective
Build and evaluate regression models to **predict `Profit`** based on sales-related and categorical features, and compare their performance using appropriate metrics and cross-validation.

### 🔹 Feature Engineering
- Target variable: **`Profit`**.
- Candidate features include:
  - Numeric: `Sales`, `Discount`, `Quantity`.
  - Categorical: `Category`, `Sub-Category`, `Region`, `Segment`.
- Categorical variables are encoded using **one-hot encoding** (via `OneHotEncoder` in a `ColumnTransformer`).
- Data is split into **training** and **test** sets (e.g., 80/20 split) to evaluate real-world generalization.

### 🔹 Models Implemented
At least two regression models are built and compared:

1. **Multiple Linear Regression**
   - Baseline model that fits a linear relationship between features and `Profit`.

2. **Ridge Regression (L2 Regularization)**
   - Penalizes large coefficients to reduce overfitting and improve stability.

3. **Lasso Regression (L1 Regularization)**
   - Encourages sparsity in coefficients (can drive some to zero), effectively performing feature selection.

All models are wrapped in **scikit-learn Pipelines** that:
- Apply preprocessing (scaling numeric features, one-hot encoding categorical features).
- Fit the corresponding regression estimator.

### 🔹 Model Evaluation
Models are evaluated using:

- **R² (Coefficient of Determination)** — measures how much variance in `Profit` is explained by the model.
- **Mean Absolute Error (MAE)** — average absolute difference between predicted and actual profit.
- **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** — penalize larger errors more strongly.
- **5-Fold Cross-Validation** — used to estimate model performance on unseen data and reduce variance from a single train/test split.

Results are collected into a comparison table (e.g., `results_df`) that includes:

- Model name.
- Test set metrics (R², MAE, MSE, RMSE).
- Average cross-validated R² and RMSE.

A simple Matplotlib bar chart is used to compare model performance visually (e.g., R² across models).

---

## 📊 Model Performance (Our Actual Results)

| Model | Test R² | Test MAE | Test MSE | Test RMSE | CV R² Mean | CV R² Std | CV RMSE Mean | CV RMSE Std |
|-------|---------|-----------|-----------|------------|--------------|-------------|----------------|----------------|
| **Linear Regression** | **0.573341** | **14.102988** | **358.573226** | **18.936030** | **0.570063** | 0.020830 | 19.399518 | 0.435027 |
| **Ridge Regression (α = 1.0)** | 0.573317 | 14.102260 | 358.593324 | 18.936561 | 0.570074 | 0.020823 | 19.399271 | 0.435145 |
| **Lasso Regression (α = 0.001)** | 0.573318 | 14.102610 | 358.592299 | 18.936533 | 0.570070 | 0.020822 | 19.399364 | 0.434977 |

## 📝 Model Evaluation Summary (Which Model Performed Best?)

Across all three regression models — **Linear Regression**, **Ridge Regression**, and **Lasso Regression** — performance was nearly identical. Each model achieved a **test R² of approximately 0.573**, with similar MAE, MSE, and RMSE values. Cross-validation results were also stable, with an average **cross-validated R² of around 0.57** and low standard deviation across folds, indicating strong generalization and no overfitting.

Although all models performed similarly, **Ridge Regression** can be considered the best overall option because it provides slightly more stable coefficients and reduces the impact of multicollinearity while maintaining equal predictive accuracy. 

Coefficient analysis revealed several key insights about profit drivers:
- **Sales** is the strongest positive predictor of profit.
- **Discount** is one of the strongest negative predictors.
- High-margin sub-categories (e.g., *Copiers*, *Furnishings*, *Binders*) contribute positively to profit.
- Low-margin sub-categories (e.g., *Tables*, *Bookcases*, *Supplies*) consistently reduce profit.

These findings confirm that profit is heavily influenced by pricing strategy, product category, and sales volume — insights that will inform the classification, clustering, and association-rule tasks in Deliverable 3.

---

## 🔍 Feature Importance (Coefficients)

### ⭐ **Strongest Positive Predictors (Increase Profit)**  
| Feature | Coefficient |
|--------|-------------|
| Sub-Category_Copiers | **+18.219** |
| Sales | **+16.245** |
| Sub-Category_Furnishings | **+14.124** |
| Sub-Category_Binders | **+12.847** |
| Sub-Category_Envelopes | **+9.000** |
| Category_Technology | **+7.786** |
| Sub-Category_Paper | **+7.119** |
| Sub-Category_Appliances | **+5.098** |

### ⚠️ **Strongest Negative Predictors (Decrease Profit)**  
| Feature | Coefficient |
|--------|-------------|
| Sub-Category_Phones | **−1.756** |
| Sub-Category_Art | **−3.418** |
| Sub-Category_Fasteners | **−4.616** |
| Sub-Category_Bookcases | **−7.704** |
| Sub-Category_Machines | **−8.377** |
| Category_Furniture | **−10.135** |
| Sub-Category_Supplies | **−11.184** |
| Sub-Category_Storage | **−12.590** |
| Discount | **−15.668** |
| Sub-Category_Tables | **−16.238** |

---
## ⚠️ Challenges Encountered & How They Were Addressed

### 1. High Dimensionality from Categorical Features
The dataset contains several categorical variables with many unique values (e.g., 17 Sub-Categories). This increases the number of features after one-hot encoding and can make models unstable.

**How it was addressed:**  
Used `OneHotEncoder(handle_unknown="ignore")` within a `ColumnTransformer` to ensure stable preprocessing and avoid errors when unseen categories appear in the test set.

---

### 2. Noise and Variability in the Profit Target
`Profit` is naturally noisy due to discounts, shipping costs, and varying product margins. This makes high R² difficult to achieve.

**How it was addressed:**  
Applied regularization models (**Ridge and Lasso**) to stabilize coefficients and reduce sensitivity to noise. While R² scores remained similar, regularization improved consistency and interpretability.

---

### 3. Impact of Outliers on Model Stability
Extreme values in `Sales`, `Profit`, `Discount`, and `Quantity` can distort regression results.

**How it was addressed:**  
Outliers were clipped during Deliverable 1 using the **IQR method**, improving model stability and reducing noise before regression modeling.

---

### 4. Ensuring Reliable Model Evaluation
A single train/test split can give misleading results due to randomness.

**How it was addressed:**  
Used **5-fold cross-validation** to compute mean and standard deviation for R² and RMSE, ensuring reliable and generalized evaluation across multiple data subsets.


## 🧠 Key Insights  
- **Discount** is one of the strongest negative drivers of profit  
- **Sales** is the strongest positive predictor  
- High-margin categories include Copiers, Technology, Envelopes, and Binders  
- Low-margin categories include Tables, Bookcases, Supplies  
- Regularized models did NOT improve results → data is stable  

---


## 📦 Repository Contents
```
MSCS_634_ProjectDeliverable2/
├─ Deliverable_2.ipynb
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ src/
│  └─ utils.py
└─ data/
   ├─ superstore.csv              # <- original data
   └─ cleaned_superstore.csv      # <- generated by notebook
```

---

## 🧰 How to Run
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

1. Put your Kaggle file (`Sample - Superstore.csv`) into the `data/` folder as either:
   - `superstore.csv`, **or**
   - keep the original name `Sample - Superstore.csv` (the notebook checks both).
2. Open **Deliverable_2.ipynb**.
3. Run all cells:
   - Deliverable 1 cells will clean the data, perform EDA, and save `cleaned_superstore.csv`.
   - Deliverable 2 cells will load the cleaned data, build regression models, and compare their performance.