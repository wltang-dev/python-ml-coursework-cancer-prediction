# 🎯 ML Coursework Supplement – Cancer PCR/RFS Prediction

This small project is a **supplementary coursework work** completed during my machine learning course in 2023.  
The main goal was to explore predictive modeling on a medical dataset using Python and common ML libraries.

This is **not my primary project** but rather an additional demonstration of practical machine learning techniques.

---

##  Prediction Objectives

- **PCR (Pathologic Complete Response) – Classification Task**  
  Predict whether a patient undergoing surgery will achieve complete tumor remission.  
  Binary outcome: PCR Positive or PCR Negative.

- **RFS (Recurrence-Free Survival) – Regression Task**  
  Estimate the time (in months or years) the patient is expected to live without signs of recurrence post-treatment.  
  Continuous numerical prediction.

---

##  Dataset Summary

- **Clinical features:**  
  Age, ER/PgR/HER2 status, Tumor stage, Histology, Chemo grade, etc.

- **MRI image features (107 total):**  
  Radiomics descriptors extracted from MRI scans, providing quantitative data on tumor shape, texture, and size.

---

## Project Structure

project/
│
├── data/
│ ├── original/ # Raw input data (.xls)
│ ├── datafeatureselection/ # Selected features
│ └── processed/ # Cleaned datasets
│
├── models/ # Trained models (.pkl)
├── results/ # Output predictions
├── src/ # Core modules
│ ├── config.py
│ ├── data_processing.py
│ ├── model_training.py
│ ├── feature_selection.py
│ └── main.py
├── tests/ # Unit tests (optional)
└── README.md

##  Techniques Used

- Missing value handling (numeric and gene-based logic)
- Feature selection (correlation + domain filtering)
- Model training with:
    - Linear Regression
    - Lasso Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - MLP (Multi-Layer Perceptron)
    - SVM
- **Model Blending** using weighted average of selected models
- Evaluation with MSE and R² metrics

---

##  Model Blending Example

```python
weights = [0.2, 0.2, 0.2, 0.2, 0.2]
models = [rf_model, xgb_model, lgbm_model, gb_model, mlp_model]

blended_preds = blended_prediction(models, X_test, weights)
mse = mean_squared_error(y_test, blended_preds)
r2 = r2_score(y_test, blended_preds)
This blending approach improved robustness and reduced variance on test predictions.

 Notes

This project was originally developed as part of a group coursework assignment.

The portion of the code uploaded here was ** written by me**, and reflects my individual contribution to the project.

While the overall structure and goals were discussed as a team, this repository contains the modules I personally implemented or significantly contributed to.

Please note that the dataset used is not publicly available and was provided for coursework purposes only.

 Disclaimer
This project is intended to showcase basic machine learning workflows as practiced in academic coursework.
It is not optimized nor intended for clinical use.

Contact
Feel free to reach out via GitHub if you want to know more about the structure or models used.