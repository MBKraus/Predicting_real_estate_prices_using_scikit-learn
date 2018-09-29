
import pandas as pd
from geopy import geocoders
from geopy.exc import GeocoderTimedOut

# Concatenate all separate data files retrieved by means of the scraper

a = pd.read_csv('AmsterdamPage1to50.csv', sep=';')
b = pd.read_csv('AmsterdamPage51to100.csv', sep=';')
c = pd.read_csv('AmsterdamPage101to155.csv', sep=';')

data = pd.concat([a, b, c,], ignore_index=True)

# Clean up data

data['rooms_new'] = data['rooms'].str[-2:]
data['zipcode_new'] = data['zipcode'].str[:4]
data['price_new'] = data['price'].str.replace(' von', '')

# Retrieve lat lon coordinates (mapped to zipcode)

g = geocoders.GoogleV3(api_key='your API-key')

latcoordinates = {}
loncoordinates = {}

for x in data['zipcode_new'].unique():
    try:
        y = x + " Amsterdam"
        location = g.geocode(y, timeout=60)
        lat = location.latitude
        latcoordinates[x] = lat
        lon = location.longitude
        loncoordinates[x] = lon
        print(y)
        #time.sleep(5)
    except GeocoderTimedOut as e:
        print("Error: geocode failed on input %s with message %s" % (y, e.message))


data["latitude"] = data["zipcode_new"].map(lambda x: latcoordinates[x])
data["longitude"] = data["zipcode_new"].map(lambda x: loncoordinates[x])

data.to_csv("Overview_new.csv", sep=";", encoding='utf-8')
