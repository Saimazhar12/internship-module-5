import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

bedrooms = np.random.randint(1, 7, n)
bathrooms = np.random.randint(1, 5, n)
sqft_living = np.random.randint(500, 6000, n)
sqft_lot = np.random.randint(1000, 20000, n)
floors = np.random.choice([1, 1.5, 2, 2.5, 3], n)
age = np.random.randint(0, 80, n)
garage = np.random.randint(0, 4, n)
location_score = np.round(np.random.uniform(1, 10, n), 1)  # 1=worst area, 10=best area

# Realistic-ish price formula with noise
price = (
    50000
    + sqft_living * 180
    + bedrooms * 8000
    + bathrooms * 12000
    + floors * 5000
    + garage * 6000
    + location_score * 15000
    - age * 900
    + sqft_lot * 1.5
    + np.random.normal(0, 25000, n)
)
price = np.clip(price, 40000, None).round(0)

df = pd.DataFrame({
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "age": age,
    "garage": garage,
    "location_score": location_score,
    "price": price,
})

df.to_csv("house_price.csv", index=False)
print(df.head())
print(df.shape)
