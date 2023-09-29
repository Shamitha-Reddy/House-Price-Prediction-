# House-Price-Prediction-
Machine Learning model to predict house price using linear regression

1. **Importing Libraries:** The code begins by importing the necessary Python libraries. These libraries are used for data manipulation, visualization, and machine learning:

   - `import pandas as pd`: Imports the Pandas library for data manipulation.
   - `import numpy as np`: Imports the NumPy library for numerical operations.
   - `import matplotlib.pyplot as plt`: Imports Matplotlib for creating plots.
   - `import seaborn as sns`: Imports Seaborn for enhanced data visualization.

2. **Loading Data:** The code loads a dataset called 'USA_Housing.csv' into a Pandas DataFrame named 'USAhousing' using the `pd.read_csv()` function. It then displays the first few rows of the DataFrame using `USAhousing.head()` and provides information about the DataFrame's structure using `USAhousing.info()` and basic statistics using `USAhousing.describe()`.

3. **Data Exploration and Visualization:**
   - `USAhousing.columns`: Lists the column names of the DataFrame.
   - `sns.pairplot(USAhousing)`: Creates a pairplot, which is a grid of scatterplots showing relationships between different numerical variables in the dataset.
   - `sns.distplot(USAhousing['Price'])`: Creates a histogram and a kernel density estimate plot for the 'Price' column, which is likely the target variable.
   - `sns.heatmap(USAhousing.corr())`: Creates a heatmap to visualize the correlation between numerical variables in the dataset.

4. **Data Preprocessing:**
   - The code selects specific columns as features (`X`) and the 'Price' column as the target variable (`y`) for the linear regression model.

5. **Data Splitting:** The dataset is split into training and testing sets using `train_test_split()` from scikit-learn. 60% of the data is used for training (`X_train` and `y_train`), and 40% is used for testing (`X_test` and `y_test`). The `random_state` parameter ensures reproducibility.

6. **Linear Regression Model:**
   - A linear regression model is created using `LinearRegression()` from scikit-learn.
   - The model is trained on the training data using `lm.fit(X_train, y_train)`. This step finds the coefficients for the linear regression equation.

7. **Model Evaluation and Prediction:**
   - `print(lm.intercept_)`: Prints the intercept (bias) of the linear regression model.
   - `coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])`: Creates a DataFrame to display the coefficients of the features.
   - `predictions = lm.predict(X_test)`: Uses the trained model to make predictions on the test data.
   - `plt.scatter(y_test, predictions)`: Creates a scatter plot to visualize the relationship between the actual target values (`y_test`) and the predicted values (`predictions`).
   - `sns.distplot((y_test - predictions), bins=50)`: Creates a histogram of the residuals (the differences between actual and predicted values).

8. **Model Evaluation Metrics:**
   - The code calculates and prints three metrics to evaluate the model's performance: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). These metrics quantify how well the model predicts the 'Price' target variable.

In summary, this code performs a linear regression analysis on the 'USA_Housing' dataset, including data exploration, model training, evaluation, and visualization of results. It also provides information about the model's coefficients and various performance metrics to assess its quality.
