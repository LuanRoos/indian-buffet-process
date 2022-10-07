import pandas as pd
import numpy as np
import argparse

def build_dataset(restaurant_idx):
    df = pd.read_csv(f'data/restaurant/restaurant-{restaurant_idx}-orders.csv',parse_dates=['Order Date'],dayfirst=True)
    
    df['Weekday'] = df['Order Date'].dt.day_of_week
    
    weather_df = pd.read_csv('data/weather/london_weather.csv')
    
    weather_df['date'] = pd.to_datetime(weather_df['date'], format='%Y%m%d')
    weather_df.drop(columns=['global_radiation', 'pressure'], inplace=True)
    
    df['date'] = df['Order Date'].dt.date
    weather_df['date'] = weather_df['date'].dt.date
    
    merge_df = pd.merge(df, weather_df, on='date', how='left')
    merge_df.drop(columns=['date'], inplace=True)
    
    merge_df.to_pickle(f'data/restaurant/processed{restaurant_idx}.pkl')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument('-d', "--data", action="store_true", help="download, process and save data to scv")
    args = parser.parse_args()
    config = vars(args)
    
    if config['data']:
        build_dataset(1)
        build_dataset(2)
        
        