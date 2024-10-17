# Data Science & Business Analytics - Prediction using Supervised ML

This project is part of my internship at **The Sparks Foundation**, where I worked on the task of predicting student scores based on the number of hours they studied. The task involves implementing **Simple Linear Regression** to predict the percentage of marks a student is expected to score based on study hours.

## Project Details

- **Domain**: Data Science & Business Analytics
- **Language**: Python
- **IDE**: Jupyter Notebook
- **Task**: Predict the percentage of marks a student will score based on the number of hours studied (Simple Linear Regression)

### Objective

The objective of this project is to predict the percentage of marks that a student will score based on the number of study hours they have undertaken. We are using a **Simple Linear Regression** model, which involves only two variables: `Hours` and `Scores`.

## Steps

1. **Data Collection**:
   The dataset is obtained from an online source and contains two columns:
   - `Hours`: The number of hours studied by a student.
   - `Scores`: The percentage score achieved by the student.

2. **Exploratory Data Analysis (EDA)**:
   Visualizing the dataset to check for patterns and relationships. We can see a positive linear relationship between the hours studied and the scores obtained.

3. **Data Preparation**:
   Splitting the dataset into training and testing data using `train_test_split()`.

4. **Model Training**:
   Using **Linear Regression** to train the model on the training data and predict scores based on the number of hours studied.

5. **Prediction**:
   Making predictions on the test set and comparing them with actual values.

6. **Model Evaluation**:
   Evaluating the performance of the model using metrics like **R2 Score**.

## Libraries Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **plotly**: For creating interactive visualizations and plotting graphs.
- **sklearn (scikit-learn)**: For building and evaluating the linear regression model.

## Code Overview

### Importing Libraries and Dataset

```python
import pandas as pd
import numpy as np
import plotly.express as px  
import plotly.graph_objects as go

url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")
```

### Data Exploration

The dataset consists of two columns: `Hours` (study hours) and `Scores` (percentage scores).

```python
data.info()
data.describe()
```

### Data Visualization

We use Plotly to visualize the relationship between hours studied and scores obtained:

```python
fig = px.scatter(data, x='Hours', y='Scores', opacity=0.65, color='Scores', title='Hours vs Percentage')
fig.show()
```

### Splitting the Data

We split the data into training and testing sets using `train_test_split`:

```python
from sklearn.model_selection import train_test_split  

X = data.iloc[:, :-1].values 
y = data.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Training the Model

We train a **Linear Regression** model using the training data:

```python
from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print("Training complete.")
```

### Predictions and Evaluation

We make predictions on the test set and compare them with the actual values:

```python
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
```

The performance of the model is evaluated using the **R2 Score**:

```python
from sklearn import metrics
print('R2 Score:', metrics.r2_score(y_test, y_pred))
```

### Example Prediction

You can also test the model with your own data. For example:

```python
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
```

---

## Results

The model has a high **R2 score** of **0.945** on the test data, indicating that the model fits the data very well and can accurately predict the student’s score based on study hours.

## Gratitude

I would like to express my sincere **gratitude** to **The Sparks Foundation** for providing me with the opportunity to work as an intern on this project. The experience has been incredibly valuable in sharpening my skills in **Data Science**, **Machine Learning**, and **Business Analytics**.

---

## License

This project is open-source and free to use. Feel free to modify the code for personal or academic purposes.
