import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, validation_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

#----------------------------------------------------------------------------------------

# Load & do first stage pre-processing (i.e. exclude non-numeric prices and convert prices to numeric variables)

file = pd.read_csv('Final.csv', sep=';')
df = pd.DataFrame(file)
df = df[df.price_new.str.contains("aanvraag") == False].dropna()
df[['price_new']] = df[['price_new']].apply(pd.to_numeric)

# exclude everything with a price above or below 3 standard deviations (i.e. outliers)

data = df[np.abs(df["price_new"]-df["price_new"].mean())<=(3*df["price_new"].std())]

# Normalise data

data = (data - data.mean()) / (data.std())

# Set x and y - and split into train and test sets

y = data.price_new
x = data.drop('price_new', axis = 1).drop('zipcode_new', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)

# Select Matplotlib style

plt.style.use('seaborn')

#----------------------------------------------------------------------------------------

### Descriptives

# Scatterplot - Asking Price (EUR) vs. Surface (m2) (NB - this plot uses non-normalised data)

plt.scatter(data['surface'], data['price_new'], alpha=0.7, label='Asking price (EUR) vs. Surface (m2)')
plt.xticks([0,100,200, 300, 400,500])
plt.yticks([0, 500000, 1000000, 1500000, 2000000, 2500000])
plt.ylabel('Asking price (EUR)')
plt.xlabel('Surface (m2)')
plt.title('Asking price (EUR) vs. Surface (m2)')
plt.text(0,2400000, ' Mean asking price (EUR) = {}'.format(round(float(data['price_new'].mean()), 2)))
plt.text(0,2300000, ' Mean surface (m2) = {}'.format(round(float(data['surface'].mean()), 2)))
plt.show()

# Histogram - Asking Price (EUR)

plt.hist(data['price_new'], bins=50, normed=False, histtype='stepfilled', alpha=0.7)
plt.title('Histogram - Asking price (EUR)')
plt.show()

# Boxplot Asking price (EUR) by Zip code + stats by Zip code

data.boxplot(by=["zipcode_new"], column=["price_new"])
plt.show()
print(data['price_new'].groupby(data['zipcode_new']).describe())

# Correlation matrix

corr = data.drop('zipcode_new', axis = 1).corr()
print(corr)

#----------------------------------------------------------------------------------------

### Ordinary Least Squares (OLS) Regression

lm = LinearRegression()
lm.fit(x_train, y_train)
predicted = lm.predict(x_test)

plt.scatter(y_test, predicted)
plt.plot([-2,6], [-2,6], "g--", lw=1, alpha=0.4)
plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.axis([-2,6,-2,6])
plt.text(-1,3, ' R-squared = {}'.format(round(float(lm.score(x_test,y_test)), 2)))
plt.text(-1,3.5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted)), 2)))
plt.title('OLS - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

print("Model coefficients: ", list(zip(list(x_test), lm.coef_)))

### 10 folds cross-validation along the previous OLS Regression

lm = LinearRegression()
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(lm, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#----------------------------------------------------------------------------------------

### KNN Regression

# GridsearchCV for KNN regression (CV=10)

parameters = {'n_neighbors':[5, 10, 15, 25, 30]}
knn = KNeighborsRegressor()
grid_obj = GridSearchCV(knn, parameters, cv=10, scoring='r2')
grid_obj.fit(x, y)

print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)
print(pd.DataFrame(grid_obj.cv_results_))

# Validation curve over different ranges of N for KNN regression with 10 fold cross-validation

param_range = tuple(list(range(1,20)))
train_scores, test_scores = validation_curve(KNeighborsRegressor(), x, y, param_name="n_neighbors", param_range=param_range,cv=10)
test_scores_mean = np.mean(test_scores, axis=1)

plt.title("Validation Curve with KNN")
plt.xlabel("KNN")
plt.ylabel("Score")
plt.plot(param_range, test_scores_mean, label="Test score",)
plt.legend(loc="best")
plt.show()

# KNN Regression with the optimal n (n_neighbors = 15)

knn = KNeighborsRegressor(n_neighbors=15)
knn.fit(x_train, y_train)
predicted_knn = knn.predict(x_test)

plt.scatter(y_test, predicted_knn)
plt.plot([-2,6], [-2, 6], "g--", lw=1, alpha=0.4)

plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.text(-1,3.5, ' R-squared = {}'.format(round(float(knn.score(x_test,y_test)), 2)))
plt.text(-1,3, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_knn)), 2)))
plt.title('KNN (15) - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

# 10 folds cross-validation along the previous KNN regression

knn = KNeighborsRegressor(n_neighbors=15)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(knn, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#-------------------------------------------------------------------------

### Lasso Regression

# GridSearchCV for Lasso Regression (CV=10)

parameters={'alpha': [0.25,1,5,10,15,20,100]}
lasso_reg = Lasso(max_iter=1500)
grid_obj = GridSearchCV(lasso_reg,parameters,cv=10, scoring = 'r2')
grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Lasso Regression along optimal alpha (0.25)

lasso = Lasso(alpha=0.25)
lasso.fit(x_train, y_train)
predicted_lasso = lasso.predict(x_test)

plt.scatter(y_test, predicted_lasso)
plt.plot([-1.5,5], [-1.5,5], "g--", lw=1, alpha=0.4)

plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.axis([-1.5,5,-1.5,5])
plt.text(-1,4, ' R-squared = {}'.format(round(float(lasso.score(x_test,y_test)), 2)))
plt.text(-1,4.5, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_lasso)), 2)))
plt.title('Lasso (Alpha - 0.25) - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

# 10 folds cross-validation along the previous Lasso regression

lasso = Lasso(alpha=0.25)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(lasso, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#-------------------------------------------------------------------------

### Ridge Regression

# GridSearchCV - Ridge Regression

parameters={'alpha': [0.1, 5,10,15,20,100]}
rdg_reg = Ridge()
grid_obj = GridSearchCV(rdg_reg,parameters,cv=10, scoring = 'r2')
grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

plt.plot(results['param_alpha'], results['mean_test_score'], 'g-', alpha=0.4)
plt.text(0,0.667, ' Best-score = {}'.format(round(float(grid_obj.best_score_), 2)))
plt.text(0,0.666, ' Optimal alpha = {}'.format(grid_obj.best_params_))
plt.xlabel("Alpha")
plt.ylabel("Mean test score")
plt.title('GridsearchCV RidgeRegressor (CV=10)')
plt.show()

# RidgeCV Regression with 10 fold cross-validation along alpha values of 0.1, 1 and 10

Ridge_CV = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=10)
Ridge_CV.fit(x_train, y_train)
predicted_Ridge_CV = Ridge_CV.predict(x_test)

plt.scatter(y_test, predicted_Ridge_CV)
plt.plot([-1,5], [-1,5], "g--", lw=1, alpha=0.4)

plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")

plt.text(-1,2.5, ' R-squared = {}'.format(round(float(Ridge_CV.score(x_test,y_test)), 2)))
plt.text(-1,3, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_Ridge_CV)), 2)))
plt.title('Ridge (Alpha = {}) - Predicted prices (EUR) vs. True prices (EUR)'.format(Ridge_CV.alpha_))
plt.show()

# 10 folds cross-validation along the previous Ridge regression

ridge = Ridge(alpha=0.1)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(ridge, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#-------------------------------------------------------------------------

### Polynomial regression (degrees = 4)

poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_poly, y, random_state = 0)

poly_reg = LinearRegression().fit(x_train1, y_train1)
predicted_poly = poly_reg.predict(x_test1)

plt.scatter(y_test1, predicted_poly)
plt.plot([-1,5], [-1,5], "g--", lw=1, alpha=0.4)
plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.axis([-2,6,-2,6])
plt.text(-1,3, ' R-squared = {}'.format(round(float(poly_reg.score(x_test1,y_test1)), 2)))
plt.text(-1,4, ' MSE = {}'.format(round(float(mean_squared_error(y_test1, predicted_poly)), 2)))
plt.title('Poly (Four degrees) - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

# 10 folds cross-validation along the previous Polynomial regression (degrees =4)

LR = LinearRegression()
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(LR, x_poly, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#-------------------------------------------------------------------------

### Support Vector Regression

# GridSearchCV - Support Vector Regression

Cs = [10, 100]
gammas = [0.001, 0.01, 0.1, 1]
parameters = {'C': Cs, 'gamma' : gammas}
grid_obj = GridSearchCV(SVR(kernel='linear'), parameters, cv=3)
grid_obj.fit(x, y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Support Vector Regression along optimal parameters

svr = SVR(kernel='linear', gamma = 0.001, C= 10)
svr.fit(x_train, y_train)
predicted_svr = svr.predict(x_test)

plt.scatter(y_test, predicted_svr)
plt.plot([-1,5], [-1,5], "g--", lw=1, alpha=0.4)

plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.text(-1,3, ' R-squared = {}'.format(round(float(svr.score(x_test,y_test)), 2)))
plt.text(-1,4, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_svr)), 2)))
plt.title('SVR (Gamma = 0.001, C=10) - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

# 10 folds cross-validation along the previous SVR

svr = SVR(kernel='linear', gamma = 0.001, C= 10)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(svr, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#------------------------------------------------------------------------------
### Decision Tree Regression

# GridSearchCV - Decision Tree Regression

tree = DecisionTreeRegressor()
parameters = {"max_depth": range(2,5), "random_state":[0], "min_samples_leaf": [6,7,8,9]}
grid_obj = GridSearchCV(estimator=tree,param_grid=parameters, cv=2, scoring='r2')
grid_fit =grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Decision Tree Regression along optimal parameters

tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=6)
tree.fit(x_train, y_train)
predicted_tree = tree.predict(x_test)

plt.scatter(y_test, predicted_tree)
plt.plot([-2, 5], [-2,5], "g--", lw=1, alpha=0.4)

plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.axis([-2,5,-2,5])
plt.text(-1,4, ' R-squared = {}'.format(round(float(tree.score(x_test,y_test)), 2)))
plt.text(-1,3, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_tree)), 2)))
plt.title('Decision Tre Regressor (max_depth =4, min_sample_leaf=8) - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

print("Feature importances: ", list(zip(list(x_test), tree.feature_importances_)))

# 10 folds cross-validation along the previous Decision Tree

tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=6)
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(tree, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#------------------------------------------------------------------------------

### RandomForest Regression

# GridSearchCV - Random Forest Regression

forest = RandomForestRegressor()
parameters = {"n_estimators": [10, 20], "max_features": [4], 'max_depth': [None, 1, 2, 3]}
grid_obj = GridSearchCV(estimator=forest, param_grid=parameters, cv=5, scoring='r2')
grid_obj.fit(x,y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Random Forest regression along optimal parameters

forest = RandomForestRegressor(n_estimators=20, max_features=4)
forest.fit(x_train, y_train)
predicted_forest = forest.predict(x_test)

plt.scatter(y_test, predicted_forest)
plt.plot([-2, 5], [-2,5], "g--", lw=1, alpha=0.4)
plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.axis([-2,5,-2,5])
plt.text(-1,4, ' R-squared = {}'.format(round(float(forest.score(x_test,y_test)), 2)))
plt.text(-1,3, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_forest)), 2)))
plt.title('Random Forest Regression (N=20,max_feat=4) - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

# 10 folds cross-validation along the previous Random Forest Regression

forest= RandomForestRegressor(n_estimators=20, max_features=4)
shuffle = KFold(n_splits=3, shuffle=True, random_state=0)
cv_scores = cross_val_score(forest, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())

#------------------------------------------------------------------------------

### Neural Network

### GridSearchCV - Neural Network MLP Regression

parameters = {'alpha':[5, 10, 100]}
MLP = MLPRegressor(hidden_layer_sizes= [3,3], random_state=0, solver='lbfgs')
grid_obj = GridSearchCV(MLP, parameters, cv=5, scoring='r2')
grid_obj.fit(x, y)

results = pd.DataFrame(grid_obj.cv_results_)

print(pd.DataFrame(results))
print("best_index", grid_obj.best_index_)
print("best_score", grid_obj.best_score_)
print("best_params", grid_obj.best_params_)

# Neural Network MLP regression along optimal parameters

neural = MLPRegressor(hidden_layer_sizes= [3,3], alpha=5, random_state=0, solver='lbfgs')
neural.fit(x_train, y_train)
predicted_neural = neural.predict(x_test)

plt.scatter(y_test, predicted_neural)
plt.plot([-2, 5], [-2,5], "g--", lw=1, alpha=0.4)

plt.xlabel("True prices (EUR)")
plt.ylabel("Predicted prices (EUR)")
plt.axis([-2,5,-2,5])
plt.text(-1,4, ' R-squared = {}'.format(round(float(neural.score(x_test,y_test)), 2)))
plt.text(-1,3, ' MSE = {}'.format(round(float(mean_squared_error(y_test, predicted_neural)), 2)))
plt.title('Neural Network MLP Regression (layer=[3.3], alpha=5) - Predicted prices (EUR) vs. True prices (EUR)')
plt.show()

# 10 folds cross-validation along the previous Neural Network MLP

MLP = MLPRegressor(hidden_layer_sizes= [3,3], alpha=5, random_state=0, solver='lbfgs')
shuffle = KFold(n_splits=10, shuffle=True, random_state=0)

cv_scores = cross_val_score(MLP, x, y, cv=shuffle)
print(cv_scores)
print(cv_scores.mean())