# Generate the Node Feature and Adjacency Matrices (csv files) for JHU time-series dataset for Michigan
import pandas as pd
import numpy as np
import csv
import os
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# Generate the V or X Matrix (Node Feature Matrix) from JHU COVID time-series for Michigan counties
'''
Extract the time series of the total number of confirmed cases/deaths in Michigan in the original data set.
parameters settings
days: the number of days collected
'''

days = 626  # Mar 15,2020 - Nov 30,2021

# confirmed covid-19 cumulative cases/deaths
input_original_data = 'time_series_covid19_confirmed_US.csv'
output_confirmed_inter = 'covid19_confirmed_83_counties_MI_inter.csv'
output_confirmed_cases_US = 'covid19_confirmed_83_counties_MI_X_matrix_final.csv'

pd.set_option("max_columns", None)
pd.set_option("max_colwidth", None)

df1 = pd.read_csv(input_original_data, header=None)

state = 'Michigan'
counties = []
for i in range(len(df1)):
    if state == df1.iloc[i, 6] and str(df1.iloc[i, 5]) != 'Unassigned' \
            and str(df1.iloc[i, 5]) != 'Michigan Department of Corrections (MDOC)' \
            and str(df1.iloc[i, 5]) != 'Out of MI' \
            and str(df1.iloc[i, 5]) != 'Federal Correctional Institution (FCI)':
        counties.append(str(df1.iloc[i, 5]))

print(counties)
print(len(counties))

df_mich = (df1[(df1[6] == 'Michigan') & (df1[5] != 'Unassigned') &
               (df1[5] != 'Federal Correctional Institution (FCI)') &
               (df1[5] != 'Michigan Department of Corrections (MDOC)') &
               (df1[5] != 'Out of MI')]).reset_index(drop=True)
pd.set_option('display.max_rows', None)

for i in range(len(counties)):
    start_time = time.time()
    county = counties[i]
    county_cases = []  # record cases for each county

    for j in range(days + 1):
        if county == df_mich.iloc[i, 5]:
            # since the actual infected from column 63 of df1
            # Also we want the daily cases from Mar 15
            # so we start from Mar 14 to include 1 diff from cumulative cases
            county_cases.append(int(df_mich.iloc[i, j + 63]))

    county_cases = np.diff(county_cases, n=1)  # to calculate daily infected cases from cumulative cases
    end_time = time.time()
    print("Time elapsed per county: ", end_time - start_time)

    with open(output_confirmed_inter, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(county_cases)
        csv_file.close()

    print(county, 'is done')

df2 = pd.read_csv(output_confirmed_inter, header=None)
data = df2.iloc[:, :].values
data = list(map(list, zip(*data)))
data = pd.DataFrame(data)
# X Matrix Generated for counties of Michigan and saved as a csv file
data.to_csv(output_confirmed_cases_US, header=0, index=0)

os.remove(output_confirmed_inter)

############################################################################################################
############################################################################################################

# Generate the W or Adjacency Matrix from Lat. and Long. of 83 MI Counties

output_W_inter = 'covid19_confirmed_83_counties_MI_W_matrix_inter.csv'

# Find the location of the county from the latitude and longitude
# location for each county
for i in range(len(counties)):
    county = counties[i]
    county_loc = []

    for j in range(len(df_mich)):  # record lat and long counties

        if county == df_mich.iloc[j, 5]:
            latitude = float(df_mich.iloc[j, 8])
            longitude = float(df_mich.iloc[j, 9])

    county_loc.append(latitude)
    county_loc.append(longitude)

    with open(output_W_inter, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(county_loc)
        csv_file.close()

    print(county, 'is done')

df2 = pd.read_csv(output_W_inter, header=None)
data = df2.iloc[:, :].values
data = list(map(list, zip(*data)))

output_W_inter_1 = 'covid19_confirmed_83_counties_MI_W_matrix_inter_1.csv'
output_W_final = 'covid19_confirmed_83_counties_MI_W_matrix_final.csv'

data = pd.DataFrame(data)
data.to_csv(output_W_inter_1, header=0, index=0)

df3 = pd.read_csv(output_W_inter_1, header=None)

vec = []
A = []
i = 0
j = 0

for i in range(len(counties)):

    for j in range(df3.shape[0]):
        A.append(df3.iloc[j, i])

    vec.append(1)

    vec[i] = A

    A = []

distA = pdist(vec, metric='euclidean')
distB = squareform(distA)

for a in range(len(counties)):
    A = []

    for b in range(len(counties)):
        A.append(distB[a, b])

    with open(output_W_final, "a") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(A)
        csv_file.close()

os.remove(output_W_inter)
os.remove(output_W_inter_1)




