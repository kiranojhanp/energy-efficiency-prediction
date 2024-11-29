# Energy Efficiency Prediction Using R

This project analyzes the **Energy Efficiency dataset** from the UCI Machine Learning Repository. The dataset is used to predict **Heating Load (Y1)** and **Cooling Load (Y2)** based on various building design parameters. The analysis includes correlation studies, model building (Linear Regression and Random Forest), and feature importance evaluation.

---

## Dataset Overview

The dataset consists of 768 observations with eight independent variables (X1-X8) and two target variables (Y1 and Y2). All variables are numeric, and the dataset contains no missing values.

### Variables:

| Variable                      | Type                  | Description                 |
| ----------------------------- | --------------------- | --------------------------- |
| X1: Relative Compactness      | Numeric (Continuous)  | Compactness of the building |
| X2: Surface Area              | Numeric (Continuous)  | Total surface area          |
| X3: Wall Area                 | Numeric (Continuous)  | Area of external walls      |
| X4: Roof Area                 | Numeric (Continuous)  | Area of the roof            |
| X5: Overall Height            | Numeric (Continuous)  | Height of the building      |
| X6: Orientation               | Numeric (Categorical) | Cardinal orientation        |
| X7: Glazing Area              | Numeric (Continuous)  | Window area                 |
| X8: Glazing Area Distribution | Numeric (Categorical) | Window distribution         |
| Y1: Heating Load              | Numeric (Continuous)  | Energy required for heating |
| Y2: Cooling Load              | Numeric (Continuous)  | Energy required for cooling |

---

## Key Features of the Analysis

1. **Data Cleaning and Preparation**:

   - The dataset is pre-cleaned with no missing values.
   - Numerical normalization was performed for consistent scaling.

2. **Correlation Analysis**:

   - A Spearman correlation matrix revealed significant relationships:
     - Strong negative correlation: X1 with X2 (-1.00) and X4 (-0.87).
     - Strong positive correlation: X4 and X5 with Y1/Y2 (~0.86).
     - Negligible correlation: X6 with all other variables.

3. **Modeling**:

   - Models used:
     - Linear Regression
     - Random Forest (500 trees)
   - Dataset split: 70% training, 30% validation.
   - Cross-validation with 10 folds ensures robust evaluation.

4. **Results**:

   - **Random Forest** outperformed Linear Regression with lower Mean Squared Errors (MSE) and higher R² values.
   - Feature importance showed that:
     - X2 (Surface Area) is crucial for Y1 (Heating Load).
     - X4 (Roof Area) is the most significant for Y2 (Cooling Load).

5. **Visualization**:
   - Prediction plots for both models highlight the alignment between actual and predicted values.
   - Feature importance plots identify the top predictors for energy loads.

---

## How to Run the Code

### Prerequisites:

- Install required R libraries: `caret`, `randomForest`, `ggplot2`, `dplyr`.

### Steps:

1. **Load the Dataset**:

   - Use `ENB2012_data.xlsx` for building design parameters and energy loads.
   - Ensure the file is placed in the same directory as the script.

2. **Run the Script**:

   - Execute the script `ICT515-final-exam.R` in RStudio or an R environment.
   - The script performs the entire analysis pipeline: loading data, preprocessing, correlation analysis, model training, and evaluation.

3. **Outputs**:
   - Correlation matrix and significance tests.
   - Prediction plots for both Y1 and Y2.
   - Feature importance plots for both models.
   - Model performance metrics (MSE and R²).

---

## Interpretation of Results

1. **Correlation Analysis**:

   - Variables X2 (Surface Area) and X4 (Roof Area) significantly impact energy loads, correlating positively with both targets.

2. **Model Performance**:

   - **Random Forest** captures non-linear relationships and outperforms Linear Regression:
     - Lower MSE.
     - R² values closer to 1.

3. **Feature Importance**:
   - Emphasizes the role of **Surface Area** (X2) and **Roof Area** (X4) in determining heating and cooling requirements.

---

## References

- Dataset: Energy Efficiency Data Set, UCI Machine Learning Repository.

---

## Author

**Kiran Ojha**
