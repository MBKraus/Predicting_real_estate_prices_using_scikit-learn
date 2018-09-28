import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn')

# Load data

file = pd.read_csv('Data.csv', sep=';')
df = pd.DataFrame(file)

def preprocessing(data):

    # Do first stage pre-processing (i.e. exclude non-numeric prices and convert prices to numeric variables)

    data = data[data.price_new.str.contains("aanvraag") == False].dropna()
    data[['price_new']] = data[['price_new']].apply(pd.to_numeric)

    # exclude everything with a price above or below 3 standard deviations (i.e. outliers)

    data = data[np.abs(data["price_new"]-data["price_new"].mean())<=(3*data["price_new"].std())]

    # Set x and y (dropping zipcode and rooms as latitude, longitude and surface pretty much capture the former)

    y = data.price_new
    X = data.drop('price_new', axis = 1).drop('zipcode_new', axis = 1).drop('rooms_new', axis = 1)

    return data, X, y

# Execute preprocessing

data, X, y = preprocessing(df)

# Descriptives

# Scatterplot - Asking Price (EUR) vs. Surface (m2)

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
