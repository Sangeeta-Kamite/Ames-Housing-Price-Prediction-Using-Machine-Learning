# Ames-Housing-Price-Prediction-Using-Machine-Learning
This project focuses on predicting house prices using the Ames Housing Dataset, which contains various property features such as lot size, number of rooms, year built, basement area, and other structural characteristics. The primary objective is to build regression models that accurately predict housing prices based on these features.

**1. Data Preprocessing & Cleaning**
   Handling Missing Data:
    - Used SimpleImputer with mean imputation for missing numerical values (e.g., Total Basement Square Footage).
      Encoded categorical variables (‘Central Air’) using Label Encoding.
   Outlier Detection:
    - Used box plots (via Seaborn and Matplotlib) to visualize and analyze outliers in numerical features.
   Feature Scaling:
    - Used RobustScaler to handle outliers and ensure consistent feature distributions.

 **2. Exploratory Data Analysis (EDA)**
    Data Inspection:
     - Checked for missing values, unique values in categorical variables, and correlation analysis.
    Feature Selection:
     - Identified relevant features by analyzing distributions and feature importance.
    Visualizations:
     - Used Seaborn and Matplotlib to create histograms, boxplots, scatter plots, and correlation heatmaps.
   
**3. Machine Learning Models for Regression**

   Linear Regression with Mini-Batch Gradient Descent
    - Implemented Mini-Batch Gradient Descent to optimize weight updates iteratively.
    - Key Functions Used:
       - Prediction function: predict()
       - Weight update function: update_weights()
       - Cost function: compute_cost()
    - Tracked Training Loss across epochs.
   
   Linear Regression Using sklearn
    - Used Scikit-learn’s LinearRegression() to fit the dataset.
    - Compared performance with Mini-Batch Gradient Descent.

   K-Nearest Neighbors (KNN) Regression
    - Determined optimal K (number of neighbors) using cross-validation (cv=5).
    - Trained KNeighborsRegressor(n_neighbors=optimal_k).
    - Plotted MSE vs. K to visualize how the number of neighbors impacts accuracy.
  
**4. Model Evaluation Metrics**
   1. R² Score (r2_score)
   2. Mean Squared Error (MSE) (mean_squared_error)
   3. Loss Function Analysis:
       - Tracked loss over iterations for Mini-Batch Gradient Descent.
       - Plotted MSE vs. Iterations for Linear Regression & KNN.

**5. Visualization & Interpretability**
    MSE Comparison Across Models:
     - Plotted bar charts for Train MSE vs. Test MSE to compare models.

    MSE vs. Iterations for Gradient Descent:
     - Visualized MSE reduction across 2000 iterations.

    Feature Importance Analysis:
     - Identified key features affecting price prediction.

**6. Outcomes**
   1. Accurate house price predictions using optimized ML models.
   2. Comparison of different regression techniques:
       - Linear Regression (Batch & Mini-Batch)
       - K-Nearest Neighbors (KNN)
           - Insights into feature importance & data distributions.


**7. Skills Demonstrated**
   1. Python Programming (pandas, numpy, matplotlib, seaborn, sklearn)
   2. Data Preprocessing (Handling missing values, Feature scaling, Encoding)
   3. Exploratory Data Analysis (EDA) (Outlier detection, Visualization)
   4. Regression Modeling (Linear Regression, Gradient Descent, KNN Regression)
   5. Model Evaluation (R² Score, MSE, Loss Function Analysis)
   6. Hyperparameter Tuning (Finding the best K for KNN)
   7. Visualization & Interpretability (Tracking MSE trends, Feature impact)
