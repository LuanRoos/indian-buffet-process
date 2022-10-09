# Parseltongues Hackathon


## Repository layout
- **data**: The data directory contains the datasets that were used for demand forecasting as well as various other data representations that were generated during modelling and data analysis.
- **product_demand_forecasting.ipynb**: This notebook contains the code that was used for demand forecasting and time series analysis, using ARIMA, XGBoost and Prophet.
- **torchcast.ipynb**: This notebook contains the code that was used to train torchcast for the forecasting of the various categories of products that are sold at the restaurant.
- __data_prep.py__: This file serves as a script to run for various data preprocessing and data analysis endpoints. For information on how to run the script, execute:
```
python data_prep.py -h
```
