"""
Rebecca Lersch
ISTA 421
Final

This script reads and cleans satellite data, then builds and fits linear and
logistic regression models. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


thresh = 9

data = pd.read_csv("SATDATA.csv")

data = data.dropna(subset=['Expected Lifetime (Years)'])

lin_features = [
    'Longitude of Geosynchronous Orbit (Degrees)',
    'Perigee (Kilometers)',
    'Eccentricity',
    'Inclination (Degrees)',
    'Period (Minutes)',
    'Launch Mass (Kilograms)'
]

data_lin = data[[
    'Longitude of Geosynchronous Orbit (Degrees)',
    'Perigee (Kilometers)',
    'Eccentricity',
    'Inclination (Degrees)',
    'Period (Minutes)',
    'Launch Mass (Kilograms)',
    'Expected Lifetime (Years)'
]]

data_lin = data_lin.dropna()

#this is a fix for some strange strings I had in my dataset within some colums
for col in lin_features:
    data_lin[col] = (data_lin[col].astype(str).str.replace(r"[^\d\.]", "", regex=True).replace("", np.nan).astype(float))
data_lin['Expected Lifetime (Years)'] = (data_lin['Expected Lifetime (Years)'].astype(str).str.extract(r'(\d+\.?\d*)')[0] .astype(float))


cat_features = [
    'Operator/Owner', 'Country of Operator/Owner', 'Users',
    'Purpose', 'Class of Orbit', 'Type of Orbit',
    'Contractor', 'Country of Contractor', 'Launch Site', 'Launch Vehicle'
]

data_log = data[[
    'Operator/Owner', 'Country of Operator/Owner', 'Users',
    'Purpose', 'Class of Orbit', 'Type of Orbit',
    'Contractor', 'Country of Contractor', 'Launch Site', 'Launch Vehicle', 
    'Expected Lifetime (Years)'
]]
data_log = data_log.dropna()

data_log['Expected Lifetime (Years)'] = pd.to_numeric(
    data_log['Expected Lifetime (Years)'], errors='coerce'
)


data_log['LongLived'] = (data_log['Expected Lifetime (Years)'] > thresh)

#turns expected lifetime into binary based on if it is a long life or not, and encodes
#categorical features for gradient decent 
data_log_encoded = pd.get_dummies(data_log[cat_features])


y_log = data_log['LongLived'].values


X_lin = data_lin[lin_features].values
y_lin = data_lin['Expected Lifetime (Years)'].values
X_lin = np.concatenate([np.ones((X_lin.shape[0], 1)), X_lin], axis = 1)  


X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin,y_lin,test_size=0.2)

beta_lin = np.linalg.inv(X_train_lin.T @ X_train_lin) @ X_train_lin.T @ y_train_lin
#I had to read documentation from numpy for the way to use the linear algebra package 
y_pred_lin = X_test_lin @ beta_lin

mse_lin = np.mean((y_test_lin - y_pred_lin)**2)
rmse_lin = np.sqrt(mse_lin)

print("\nLinear Regression")
print("Coefficients:", beta_lin)
print(f"MSE: {mse_lin:.4f}, RMSE: {rmse_lin:.4f}")


lin_coef_abs = np.abs(beta_lin[1:])
most_sig_lin_idx = np.argmax(lin_coef_abs)
print(f"Most significant feature (Linear): {lin_features[most_sig_lin_idx]}")

def sigmoid(z):
    """
    This is the sigmoid function which makes out logit into a probability

    Parameters
    ----------
    z : float
        Logit.

    Returns
    -------
    Float
        The probility.

    """
    z = np.array(z)
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, i=10000):
    """
    This is the logistic regression model

    Parameters
    ----------
    X : array
        The X train.
    y : array
        Y train
    i : int, optional
        how many iterations.

    Returns
    -------
    theta : TYPE
        theta.

    """
    n, n_features = X.shape
    theta = np.zeros(n_features)
    for j in range(i):
        z = X @ theta #logit
        p = sigmoid(z) #converts logit to prob
        gradient = (1/n) * X.T @ (p - y) #finds the gradient
        theta -=  gradient
    return theta


X_log_np = data_log_encoded.values
X_log_np = np.concatenate([np.ones((X_log_np.shape[0], 1)), X_log_np], axis = 1)  
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log_np, y_log)


theta_log = logistic_regression(X_train_log, y_train_log, i=10000)
y_prob_log = sigmoid(X_test_log @ theta_log)
y_pred_log = (y_prob_log >= 0.5)

accuracy = np.mean(y_pred_log == y_test_log)
precision = np.sum((y_pred_log == 1) & (y_test_log == 1)) / max(np.sum(y_pred_log == 1), 1)
recall = np.sum((y_pred_log == 1) & (y_test_log == 1)) / max(np.sum(y_test_log == 1), 1)

print("\nLogistic Regression")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


log_coef_abs = np.abs(theta_log[1:])
log_feat_names = data_log_encoded.columns.tolist()
most_sig_log = np.argmax(log_coef_abs)
print(f"Most significant feature (Logistic): {log_feat_names[most_sig_log]}")


plt.figure()
plt.scatter(y_test_lin, y_pred_lin, color='rebeccapurple')
plt.plot([y_test_lin.min(), y_test_lin.max()],
         [y_test_lin.min(), y_test_lin.max()], color='red', lw=2)
plt.xlabel("Actual Expected Lifetime")
plt.ylabel("Predicted Expected Lifetime")
plt.title("Linear Regression")
plt.show()


plt.figure()
plt.hist(y_prob_log, color='rebeccapurple')
plt.xlabel("Predicted Probability of LongLived")
plt.ylabel("Count")
plt.title("Logistic Regression")
plt.show()
