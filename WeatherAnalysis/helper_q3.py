import numpy as np
import pandas as pd


def clean_years_happiness_data(excel_file):
    happiness_data = pd.read_excel(excel_file,sheet_name="Happiness",skiprows=1)
    happiness_data = happiness_data.iloc[:,[1,2,3,8]]

    # # rename_columns 
    happiness_data = happiness_data.rename(columns={happiness_data.columns[0]: 'area_names',
                                                    happiness_data.columns[1]: 'county',
                                                    happiness_data.columns[2]: 'county_town',
                                                    happiness_data.columns[3]:'average_happiness_rating',
                                                    })

    happiness_data = happiness_data.iloc[4:]
    happiness_data.reset_index().drop('index',axis=1)
    happiness_data['average_happiness_rating'] = pd.to_numeric(happiness_data['average_happiness_rating'], errors='coerce')
    happiness_data = happiness_data.dropna(subset=['average_happiness_rating'])
    happiness_data['area_names'] = happiness_data['area_names'].replace('NORTHERN IRELAND5', 'NORTHERN IRELAND')
    for col in ['area_names', 'county', 'county_town']:
        happiness_data[col] = happiness_data[col].str.strip()

    label_map = {
        (0, 4): 'low',
        (5, 6): 'medium',
        (7, 8): 'high',
        (9, 10): 'very high'
    }

    happiness_data['label'] = happiness_data['average_happiness_rating'].apply(\
        lambda x: next((v for k, v in label_map.items() if round(x) >= k[0] and round(x) <= k[1]), None))

    happiness_data['area_names'] = happiness_data['area_names'].fillna(method='ffill')
    
    return happiness_data 


def get_station_and_happiness_data_for_a_particular_year(happiness_data):
    
    sta_county = pd.read_excel('stations_counties.xlsx')
    happiness_data = happiness_data.fillna(value="None")

    # Define a list of column names to use later
    ncols = ['area_names', 'county', 'county_town', 'average_happiness_rating', 'label']

    # Iterate over each row of the DataFrame
    for index, row in sta_county.iterrows():
        # Initialize a variable to hold the happiness data
        data = None
        
        # Extract the region name from the current row and clean it up
        region = row['Region'].upper().replace('ENGLAND', '').strip()
        region = region.replace('OF','').strip()
        
        # Check if the region name is in the list of area names in the happiness data
        if region in happiness_data['area_names'].values:
            # Extract the county name from the current row and convert it to lowercase
            county = row['County'].lower()
            # Search for an exact match of the county name in the list of counties in the happiness data
            data = happiness_data[happiness_data['county'].str.lower() == county]
        
            if data.empty:
                # If an exact match is not found, search for a partial match in the county column
                data = happiness_data[happiness_data['county'].str.lower().str.contains(county)]
                
                if data.empty:
                    # If a partial match is not found in the county column, search for an exact match in the county town column
                    data = happiness_data[happiness_data['county_town'].str.lower() == county]
                    
                    if data.empty:
                        # If an exact match is not found in the county town column, search for a partial match in the county town column
                        data = happiness_data[happiness_data['county_town'].str.lower().str.contains(county)]
            
            # If a match is found, extract the corresponding data from the happiness data
            if not data.empty:
                # Iterate over each column name in the list
                for ncol in ncols:
                    # Extract the corresponding value and store it in the current row and column
                    sta_county.loc[index, ncol] = data.iloc[0][ncol]
            else:
                # If a match is not found, search in the area names column
                data = happiness_data[happiness_data['area_names'].str.upper() == region]
                # If a match is found, extract the corresponding data from the happiness data
                if not data.empty:
                    # Iterate over each column name in the list
                    for ncol in ncols:
                        # Extract the corresponding value and store it in the current row and column
                        sta_county.loc[index, ncol] = data.iloc[0][ncol]
                        # print(ncol, sta_county.loc[index, ncol], data.iloc[0][ncol])
                else:
                    # If a match is still not found, store a null value in the current row and a new column with "col_" prefix and the original column name
                    for ncol in ncols:
                        sta_county.loc[index, ncol] = pd.NA
                        
    # sta_county['year'] = year 
    return sta_county


def full_happiness_data(df, start_year, end_year):
    
    years = range(start_year, end_year+1)
    months_1 = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    months_2 = [1, 2, 3]
    
    li = []
    for index, row in df.iterrows():
        for year in years:
            months = months_1 if year == start_year else months_2
            for month in months:
                new_dict = { 'year': year,
                            'month' : month,
                            'station' :row['Station'],
                            'area_names':row['area_names'],
                            'orig_county':row['County'],
                            'region' : row['Region'],
                            'county': row['county'],
                            'county_town':row['county_town'],
                            'average_happiness_rating':row['average_happiness_rating'],
                            'label':row['label']
                    
                }
                li.append(new_dict)
    
    return pd.DataFrame(li)

