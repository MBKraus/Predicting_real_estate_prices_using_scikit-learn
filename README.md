# Predicting Amsterdam house / real estate prices using Ordinary Least Squares-, XGBoost-, KNN-, Lasso-, Ridge-, Polynomial-, Random Forest-, and Neural Network MLP Regression (via scikit-learn)

## Approach:

- load Pandas DataFrame containing (Dec-17) housing data retrieved by means of the [following scraper](https://github.com/Weesper1985/Funda-Scraper), supplemented with longitude and latitude coordinates mapped to zip code (via [GeoPy](https://geopy.readthedocs.io/en/1.10.0/#)
- do some simple data exploration / visualisation
- remove non-numeric data, NaNs, and outliers (everything above 3 x standard dev of y)
- define explanatory variables (surface,latitude,and longitude) and independent variable (price EUR)
- split the data in train and test sets (+ normalise independent variables where required) 
- find the optimal model parameters using [scikit-learn](http://scikit-learn.org/stable/)'s GridSearchCV
- fit the model using GridSearchCV's optimal parameters
- evaluate estimator performance by means of 5 fold 'shuffled' nested cross-validation
- predict cross validated estimates of y for each data point and plot on scatter diagram vs true y

## Packages required

- [Python 3.7.0](https://www.python.org/downloads/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://docs.scipy.org/doc/)
- [scikit-learn](http://scikit-learn.org/stable/)

## Scores (5 fold nested 'shuffled'cross-validation - Rsquared)

**1. XGBoost Regression**											                                            
  * Parameters: max_depth: 5, min_child_weight: 6, gamma: 0.01, colsample_bytree: 1, subsample: 0.7
  * Score: 0.887	

**2. Random Forest Regression**        									                                   
  * Parameters: max_depth: 6, max_feat: None, n_estimators: 10
  * Score:  0.839

**3. Polynomial Regression**                                							
  * Parameters: degrees: 2
  * Score: 0.731
  
**4. Neural Network MLP Regression** 				     					
  * Parameters: act: relu, alpha: 0.01, hidden_layer_size: (10,10), learning_rate: invscal
  * Score: 0.715
  
**5. KNN Regression**                                      							
  * Parameters: n_neighbours: 10
  * Score: 0.711
  
**6. Ordinary Least-Squares Regression**                                    				
  * Parameters: None
  * Score: 0.694
  
**7. Ridge Regression** 		                                        				
  * Parameters: alpha: 0.01
  * Score: 0.694
  
**8. Lasso Regression**                                        	 					
  * Parameters: alpha 0.01
  * Score: 0.693

### Sample data input (Pandas DataFrame)
```
   surface  rooms_new  zipcode_new  price_new   latitude  longitude
0    138.0        4.0         1060     420000  40.804672 -73.963420
1    130.0        5.0         1087     550000  52.355590   5.000561
2    116.0        5.0         1061     425000  52.373044   4.837568
3     92.0        5.0         1035     349511  52.416895   4.906767
4    127.0        4.0         1013    1050000  52.396789   4.876607
```

#### Scatter plot - Surface vs. Asking Price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/Scatter.png)

#### XGBoost - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/XGB.png)

#### Random Forest - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/Forest.png)

#### Polynomial - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/Poly.png)

#### Neural Network MLP - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/MLP.png)

#### KNN - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/KNN.png)

#### OLS - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/OLS.png)

#### Lasso - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/Lasso.png)

#### Ridge - Predicted prices vs. True price (EUR)

![alt text](https://github.com/Weesper1985/Predicting_real_estate_prices_using_scikit-learn/blob/master/Ridge.png)

