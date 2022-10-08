import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def build_dataset(restaurant_idx):
    df = pd.read_csv(f'data/restaurant/restaurant-{restaurant_idx}-orders.csv',parse_dates=['Order Date'],dayfirst=True)
    df = df.loc[df['Order Date'] >= '2016-07-18']
    df['Total Price'] = df['Quantity']*df['Product Price']   
    df['year'] = df['Order Date'].dt.year
    
    df['Weekday'] = df['Order Date'].dt.day_of_week
    
    weather_df = pd.read_csv('data/weather/london_weather.csv')
    
    weather_df['date'] = pd.to_datetime(weather_df['date'], format='%Y%m%d')
    weather_df.drop(columns=['global_radiation', 'pressure'], inplace=True)
    
    df['date'] = df['Order Date'].dt.date
    weather_df['date'] = weather_df['date'].dt.date
    
    merge_df = pd.merge(df, weather_df, on='date', how='left')
    merge_df.drop(columns=['date'], inplace=True)
    
    merge_df.to_pickle(f'data/restaurant/processed{restaurant_idx}.pkl')


def gen_adjacency(df, num, uniq_items):
    dim = len(uniq_items)
    adj = np.zeros((dim, dim))
    item_lookup = {item: idx for idx, item in enumerate(uniq_items)}
    lookup = lambda x : item_lookup[x]
    
    order_ids = set(df['Order Number' if num == 1 else 'Order ID'])
    
    for oidx in order_ids:
        order_its = df[df['Order Number' if num == 1 else 'Order ID'] == oidx]['Item Name'].tolist()

        order_its = list(map(lookup, order_its))
        
        for i in order_its:
            for j in order_its:
                if i == j:
                    continue
                adj[i, j] += 1      
    
    np.save(f'data/restaurant/adj{num}.npy', adj)
    return adj

def find_popularity(adj, uniq_items, k):
    items_pop = np.sum(adj, axis=1)
    sorted_items = np.flip(np.argsort(items_pop))
    pop_items = []
    for i in range(k):
        pop_items.append(uniq_items[sorted_items[i]])
    return pop_items

def find_profitable_items(df):
    uniq_items = list(set(df['Item Name']))
    item_df = df.groupby('Item Name').agg(price_sum = ('Total Price', 'sum'))
    item_df = item_df.sort_values(by=['price_sum'], ascending=[False])
    item_df = item_df.reset_index()
    
    total_price = df['Total Price'].sum()
    part_price = 0
    for idx in item_df.index:
        part_price += item_df['price_sum'][idx]
        frac = part_price/total_price
        if frac > 0.75:
            cutoff = idx + 1
            break
    
    item_df['Item Name'][:cutoff].tolist()
    return item_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument('-d', "--data", action="store_true", help="download, process and save data to scv")
    parser.add_argument('-a', "--adj", action="store_true", help="generate item adjacency matrix")
    parser.add_argument('-po', "--pop", action="store_true", help="find ci most popular items for each restaurant")
    parser.add_argument('-pr', "--prof", action="store_true", help="find most profitable items for each restaurant.")
    parser.add_argument('-ci', "--count_items", type=int, help="number of popular items")
    args = parser.parse_args()
    config = vars(args)
    
    if config['data']:
        build_dataset(1)
        build_dataset(2)
    
    if config['adj']:
        df = pd.read_pickle('data/restaurant/processed1.pkl')
        uniq_items = list(set(df['Item Name']))
        df2 = pd.read_pickle('data/restaurant/processed2.pkl')
        uniq_items2 = list(set(df2['Item Name']))
        adj1 = gen_adjacency(df, 1, uniq_items)
        adj2 = gen_adjacency(df2, 2, uniq_items2)
        
    if config['pop']:
        if not 'df' in locals():
            df = pd.read_pickle('data/restaurant/processed1.pkl')
        if not 'df2' in locals():
            df2 = pd.read_pickle('data/restaurant/processed2.pkl')
        if not 'adj1' in locals():
            adj1 = np.load('data/restaurant/adj1.npy')
        if not 'adj2' in locals():
            adj2 = np.load('data/restaurant/adj2.npy')
        if not 'uniq_items' in locals():
            uniq_items = list(set(df['Item Name']))
        if not 'uniq_items2' in locals():
            uniq_items2 = list(set(df2['Item Name']))
        
        k = 20 #config['count_items']
        pop_items1 = find_popularity(adj1, uniq_items, k)
        pop_items2 = find_popularity(adj2, uniq_items2, k)
        
        print('==============================Restaurant 1===============================================')
        print(pop_items1)
        print('==============================Restaurant 2===============================================')
        print(pop_items2)
        
    if not config['prof']:
        if not 'df' in locals():
            df = pd.read_pickle('data/restaurant/processed1.pkl')
        if not 'df2' in locals():
            df2 = pd.read_pickle('data/restaurant/processed2.pkl')
            
        prof1 = find_profitable_items(df)
        prof2 = find_profitable_items(df2)
        
        print('==============================Restaurant 1===============================================')
        print(prof1)
        print('==============================Restaurant 2===============================================')
        print(prof2)