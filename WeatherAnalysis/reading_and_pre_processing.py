import requests
import numpy as np
import pandas as pd
import re
import sklearn
from scipy.interpolate import griddata

# def read_station_file(path_to_station_file):
#     with open("stations.txt",'r') as f:
#         stations = f.readlines()
#     stations = [i.strip() for i in stations]
#     return stations

stations = ['aberporth', 'armagh', 'ballypatrick', 'bradford', 'braemar',
       'camborne', 'cambridge', 'cardiff', 'chivenor', 'cwmystwyth',
       'dunstaffnage', 'durham', 'eastbourne', 'eskdalemuir', 'heathrow',
       'hurn', 'lerwick', 'leuchars', 'lowestoft', 'manston', 'nairn',
       'newtonrigg', 'oxford', 'paisley', 'ringway', 'rossonwye',
       'shawbury', 'sheffield', 'southampton', 'stornoway',
       'suttonbonington', 'tiree', 'valley', 'waddington', 'whitby',
       'wickairport', 'yeovilton']

# get a particular station's data
def get_station_data(station:str):
    url:str = f'https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/'
    try:
        res = requests.get(f'{url}{station}data.txt')
        data = res.text
        return  data
        # print(data)
    except requests.exceptions.RequestException as e:
        print(f"Error:{e}")
        
# get full station data and put in dataframe
def get_all_data_and_preprocess():
    full_data = pd.DataFrame()
    for st in stations:
        data_str = get_station_data(st)
        lines = data_str.split('\n')
        lines = [line.strip()for line in lines]
        pattern = r'Lat\s+(-?\d+\.\d+)\s+Lon\s+(-?\d+\.\d+)'
        
        lat_long = [(float(re.search(pattern,line).group(1)),float(re.search(pattern,line).group(2))) 
                for line in lines if re.search(pattern,line)]

        start_index = lines.index('yyyy  mm   tmax    tmin      af    rain     sun')
        numerical_data = lines[start_index:]
        numerical_data = [s.split() for s in numerical_data]
        data = [d for d in numerical_data[2:]]
        
        #dropping the provisional or any other columns after sun and extending any data which doesnot have 7 values wiht 'None' values
        data_ = [x[0:7]  if len(x) > 7 else x +((['None'] * (7-len(x)))) for x in data]
        
        df = pd.DataFrame(data_, columns=['year', 'month', 'tmax(degC)', 'tmin(degC)', 'af(days)', 'rain(mm)', 'sun(hours)'])
        df['station'] = st
        df['latitude'] = lat_long[0][0]
        df['longitude'] = lat_long[0][1]
        
        # # replacing any missing values('---') and 'None' values with pd.Na 
        # df.replace(['---','None'],pd.NA,inplace=True)
        
        # and extracting all float from float 
        # with words attached(like all # converting num* and num# to just num string)
        df = df.applymap(lambda x: re.findall(r'\d+\.\d+|\d+', str(x))[0] if (re.findall(r'\d+\.\d+|\d+', str(x))) else x)
        full_data = pd.concat([full_data, df])
    
    return full_data

# clean the data 
def clean_further(data:pd.DataFrame):
    # removing rows whose year and month are integers
    data[['year','month','af(days)']] = data[['year','month','af(days)']].apply(pd.to_numeric, errors='coerce')

    # remove rows where either year or month column value is null
    data.dropna(subset=['year', 'month'], inplace=True)
    
    # converting columns to float
    columns = ['tmax(degC)', 'tmin(degC)', 'rain(mm)', 'sun(hours)', 'latitude', 'longitude']
    data[columns] = data[columns].apply(pd.to_numeric, errors='coerce').astype(float)
    
    data['year'] = data['year'].astype(int)
    data['month'] = data['month'].astype(int)
    
    return 

# extend the data so that each station has data till March 2023
def extend_data(data:pd.DataFrame):
     # create an empty DataFrame to store the extended data
    extend_data = pd.DataFrame(columns=data.columns)
    
    for station in stations:
        # get the latest year and month for this station
        latest_year = data.loc[data['station'] == station].tail(1)['year'].values[0]    
        latest_month = data.loc[data['station'] == station].tail(1)['month'].values[0] 
        
        # if the latest data is before March 2023, extend the data
        if (latest_year < 2023) or (latest_year == 2023 and latest_month < 3):
            # create a DataFrame with missing values for each missing month and year up to March 2023
            end_date = '2023-03'
            date_range = pd.date_range(start=f'{latest_year}-{latest_month}', end=end_date, freq='MS')

            missing_data = pd.DataFrame({
            'year': date_range.year,
            'month': date_range.month
            })
            missing_data[['tmax(degC)', 'tmin(degC)', 'af(days)', 'rain(mm)', 'sun(hours)']] = np.nan
            missing_data['station'] = station 
            missing_data['latitude'] = data.loc[data['station'] == station, 'latitude'].iloc[0]
            missing_data['longitude'] = data.loc[data['station'] == station, 'longitude'].iloc[0]
            # print(missing_data)
            # append the missing data to the extended data DataFrame
            extend_data = pd.concat([extend_data, missing_data.iloc[1:]], ignore_index=True)
        
        # append the existing data to the extended data DataFrame
        station_data = data.loc[data['station'] == station].copy()
        extend_data = pd.concat([ station_data, extend_data], ignore_index=True)

    # # sort the extended data by station and date
    extend_data = extend_data.sort_values(by=['station', 'year', 'month']).reset_index(drop=True)
    return extend_data

# interpolate data to handle missing values
def interpolate_extended_data(extended_data:pd.DataFrame):
    data_to_interpolate = extended_data.copy(deep=True)

    # Define columns for spatio-temporal interpolation
    spatiotem_cols = ['tmax(degC)', 'tmin(degC)', 'rain(mm)']
    
    # Define columns for temporal interpolation
    temporal_cols = ['af(days)', 'sun(hours)']

    # Perform spatiotemporal interpolation
    for col in  spatiotem_cols:
        # Create grid for interpolation
        xi = np.array((data_to_interpolate['longitude'].values, data_to_interpolate['latitude'].values,
                    data_to_interpolate['year'].values, data_to_interpolate['month'].values), dtype=np.float64).T
        yi = data_to_interpolate[col].values
        
        # Find NaN values to interpolate
        nan_idxs = pd.isna(yi)

        # Perform interpolation
        zi = griddata(xi[~nan_idxs], yi[~nan_idxs], xi[nan_idxs], method='linear')

        # Update the weather data with interpolated values
        data_to_interpolate.loc[nan_idxs, col] = zi

    # Perform temporal interpolation
    for col in temporal_cols:
        
        # Find NaN values to interpolate
        nan_idxs = pd.isna(data_to_interpolate[col])

        # Perform interpolation
        data_to_interpolate[col] = data_to_interpolate[col].interpolate(method='linear')

    # Save interpolated weather data
    data_to_interpolate.to_csv('interpolated_weather_data.csv', index=False)
    
    return data_to_interpolate  

# align the dta by selecting common years
def align_data(interpolated_df:pd.DataFrame):
    # count the number of station for a particular year
    year_counts = interpolated_df.groupby('year')['station'].nunique()
    # print(dict(year_counts))
    
    # Find years where all stations have data
    common_years = year_counts[year_counts == year_counts.max()].index.tolist()

    # Filter weather data to only include common years
    aligned_data = interpolated_df[interpolated_df['year'].isin(common_years)]
    
    aligned_data.to_csv("cleaned_aligned_data.csv",index=False)

# funcion to verify if aligned data has no nans 
# and if for each station each year aside from 2023 has 12 months
def check_if_all_align_data_function_works():
    aligned_data = pd.read_csv("cleaned_aligned_data.csv")
    
    station_list = np.unique(aligned_data['station'].values) 
    
    years = np.unique(aligned_data['year'])
    # Create an empty dictionary to store the results
    results_dict = {'Year': np.unique(aligned_data['year'])}
    # Loop over the stations
    for station in station_list:
        station_data = aligned_data[aligned_data['station'] == station]
        station_month_counts = station_data.groupby('year')['month'].nunique().to_dict()
        station_month_counts_list = [station_month_counts.get(year, 0) for year in years]
        results_dict[station] = station_month_counts_list

    # Create a dataframe from the dictionary
    results_df = pd.DataFrame(results_dict)

    # Add a column indicating if all stations have 12 months for each year
    results_df['All Stations'] = results_df.apply(lambda row: all(count == 12 for count in row[1:-1]), axis=1)
    has_nan = aligned_data.isna().any().any()
    results_df.to_excel("validation.xlsx",index=False)
    return has_nan 
    

weather_data = get_all_data_and_preprocess()
clean_further(weather_data)
extended_data = extend_data(weather_data)
interpolated_data = interpolate_extended_data(extended_data)
aligned_data = align_data(interpolated_data)

# print(check_if_all_align_data_function_works())