from helper_q3 import *
import pandas as pd
import numpy as np
import seaborn as sns


#Read the aligned data 
aligned_data = pd.read_csv('cleaned_aligned_data.csv')

happiness_2014_2015 = clean_years_happiness_data('./happiness_data/happiness-2014-2015.xls')
happiness_2013_2014 = clean_years_happiness_data('./happiness_data/happiness-2013-2014.xls')
happiness_2012_2013 = clean_years_happiness_data('./happiness_data/happiness-2012-2013.xls')
happiness_2011_2012 = clean_years_happiness_data('./happiness_data/happiness-2011-2012.xls')

happi_2014_2015 = get_station_and_happiness_data_for_a_particular_year(happiness_2014_2015)
happi_2013_2014 = get_station_and_happiness_data_for_a_particular_year(happiness_2013_2014)
happi_2012_2013 = get_station_and_happiness_data_for_a_particular_year(happiness_2012_2013)
happi_2011_2012 = get_station_and_happiness_data_for_a_particular_year(happiness_2011_2012)

full_happy_data = pd.concat([full_happiness_data(happi_2011_2012,2011,2012),
                            full_happiness_data(happi_2012_2013,2012,2013),
                            full_happiness_data(happi_2013_2014,2013,2014),
                            full_happiness_data(happi_2014_2015,2014,2015)  
])

# print(full_happy_data)

full_happy_data.sort_values(['year','month','station']).reset_index()

data_weather = aligned_data.copy(deep=True)

cols = ['area_names', 'orig_county', 'region', 'county', 'county_town', 'average_happiness_rating', 'label']

# Merge the original dataframe with the new dataframe
df_merged = pd.merge(data_weather, full_happy_data, on=['year', 'month','station'], how='left')

# Forward fill the NaN values from 1980 Jan to 2011 March with 2011 April value
df_merged.loc[df_merged['year'] < 2011, cols] = np.nan
df_merged[cols] = df_merged[cols].fillna(method='ffill')

# Backward fill the NaN values from 2015 April to 2023 March with 2015 March value
df_merged.loc[df_merged['year'] > 2015, cols] = np.nan
df_merged[cols] = df_merged[cols].fillna(method='bfill')

# print('Extended and filled dataframe:')
df_merged.sort_values(['year','station'])

df_merged[df_merged['station'] == 'aberporth'].to_excel("aberport.xlsx")

# correlation matrix 

dt = df_merged[['tmax(degC)','tmin(degC)',	'af(days)', 'rain(mm)','sun(hours)', 'average_happiness_rating']]

corr_mat = dt.corr()
print(corr_mat)
# sns.heatmap(corr_mat, annot=True, cmap='coolwarm')

