
# Diabetes Risk Prediction with ML

This project is developed to **predict the risk of diabetes** using **machine learning (ML)** algorithms.  
Various classification algorithms have been trained, compared, and the best-performing model has been detailed.

## Table of Contents
- [About the Project](#about-the-project)
- [Technologies Used](#technologies-used)
- [Installation and Usage](#installation-and-usage)
- [Modeling Process](#modeling-process)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About the Project
In this study, using diabetes data:
- Data preprocessing (Outlier removal, standardization),
- Comparison of different machine learning algorithms,
- Model selection and hyperparameter optimization,
- Prediction using real-world data  
steps have been carried out.

## Technologies Used
- Python
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Scikit-learn

## Installation and Usage
To run this project on your local machine:

1. Clone the repository:
    ```bash
    git clone https://github.com/username/Diabetes_risk_prediction_with_ML.git
    ```

2. Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3. Run the Python script:
    ```bash
    python project_name.py
    ```

> Note: The "diabetes.csv" dataset should be present in the project directory.

## Modeling Process

- **Data Visualization:**  
  - Pairplot and Correlation Heatmap were used to analyze the relationships within the data.

- **Outlier Detection:**  
  - Outliers were detected and removed using the IQR method.

- **Data Standardization:**  
  - Features were scaled using `StandardScaler`.

- **Model Training:**  
  - The following algorithms were evaluated using 10-fold cross-validation:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - SVC
    - AdaBoost
    - Gradient Boosting
    - Random Forest

- **Model Comparison:**  
  - Boxplot graphs were used to visualize the accuracy scores of all algorithms.

- **Hyperparameter Tuning:**  
  - GridSearchCV was used to determine the best hyperparameters for the Decision Tree model.

- **Testing Results:**  
  - The best model was used to compute the confusion matrix and classification report.

- **Prediction on New Data:**  
  - The trained model made a diabetes risk prediction on new individual data.

## Results
- The **Decision Tree Classifier**, optimized with GridSearch, provided the best results.
- Accuracy and other metrics of the model were reported.

## Contributing
Contributions are always welcome!  
You can open an issue or create a pull request.

## License
This project is licensed under the MIT License.
