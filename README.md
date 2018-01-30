
## Predicting Amsterdam house / real estate prices using Linear Regression, KNN-, Lasso-, Ridge-, Polynomial-, Support Vector (SVR)-, Decision Tree-, Random Forest-, and Neural Network MLP Regression.

### Approach:

- load Pandas DataFrame containing (Dec-17) housing data retrieved by means of the [following scraper](https://github.com/Weesper1985/Funda-Scraper), supplemented with longitude and latitude coordinates mapped to zip code (via [GeoPy](https://geopy.readthedocs.io/en/1.10.0/#)
- do some simple data exploration / visualisation
- remove non-numeric data, NaNs, outliers and normalise data
- define explanatory variables (surface, rooms, latitude, longitude) and independent variable (price EUR)
- split the data in train and test set for later usage
- find the optimal model parameters using [scikit-learn](http://scikit-learn.org/stable/)'s GridSearchCV
- fit the model using GridSearchCV's optimal parameters
- evaluate estimator performance by means of 10 fold 'shuffled' cross-validation

### Packages required

- [Python 3.5.1](https://www.python.org/downloads/release/python-351/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://docs.scipy.org/doc/)
- [scikit-learn](http://scikit-learn.org/stable/)

### Results along (Dec-17) Amsterdam house / real estate price data retrieved by means of the [following scraper](https://github.com/Weesper1985/Funda-Scraper)

#### Sample data input (Pandas DataFrame)
```
   surface  rooms_new  zipcode_new  price_new   latitude  longitude
0    138.0        4.0         1060     420000  40.804672 -73.963420
1    130.0        5.0         1087     550000  52.355590   5.000561
2    116.0        5.0         1061     425000  52.373044   4.837568
3     92.0        5.0         1035     349511  52.416895   4.906767
4    127.0        4.0         1013    1050000  52.396789   4.876607
```
#### Scores (10 fold 'shuffled' cross-validation - Rsquared)

- Random Forest Regression (n_estim=20, max_depth= None, max_feat=4}    0.866
- Polynomial Regression (degrees =4)                                    0.810
- Decision Tree Regression (max_depth=4, min_samples_leaf=6)            0.737
- Neural Network MLP Regression (layer =[3,3], alpha=5, solv=lbfgs)     0.721
- KNN Regression (n-neighbors = 15)                                     0.704
- Ordinary Least-Squares Regression:                                    0.695
- Ridge Regression (alpha = 0.1)                                        0.695
- Support Vector Regression (kernel='linear', gamma = 0.001, C= 10)     0.690
- Lasso Regression (alpha = 0.25)                                       0.614


#### Scatter plot - Surface vs. Asking Price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### OLS - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### KNN Regression - Validation curve

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### Lasso Regression (Alpha = 0.25) - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### Ridge Regression - Mean Test Score vs. Alpha

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### Polynomial Regression (degrees = 4) - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### Random Forest Regression (n_estim=20, max_depth= None, max_feat=4) - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### Support Vector Regression (kernel='linear', gamma = 0.001, C= 10) - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### Decision Tree Regression (max_depth=4, min_samples_leaf=6)  - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)

#### Neural Network MLP Regression (layer =[3,3], alpha=5, solv=lbfgs) (max_depth=4, min_samples_leaf=6)  - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Python_Portfolio__VaR_Tool/blob/master/Tab11.png)
