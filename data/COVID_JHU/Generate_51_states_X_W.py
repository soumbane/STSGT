# Generate the Node Feature and Adjacency Matrices (csv files) for JHU time-series dataset for US states
import pandas as pd
import numpy as np
import csv
import os
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# Generate the V or X Matrix (Node Feature Matrix) from JHU COVID time-series
'''
Extract the time series of the total number of confirmed cases/deaths in each state in the original data set.
parameters settings
days: the number of days collected
'''

days = 626  # Mar 15,2020 - Nov 30,2021

# confirmed covid-19 cumulative cases/deaths
input_original_data = 'time_series_covid19_confirmed_US.csv'
# input_original_data = 'time_series_covid19_deaths_US.csv'

output_confirmed_inter = 'covid19_confirmed_US_51_states_inter.csv'
# output_confirmed_inter = 'covid19_deaths_US_51_states_inter.csv'

output_confirmed_cases_US = 'covid19_confirmed_US_51_states_X_matrix_final.csv'
# output_confirmed_cases_US = 'covid19_deaths_US_51_states_X_matrix_final.csv'

# We included Washington D.C. as a node for the graph similar to the states of US (51 nodes of graph)
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
          'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
          'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
          'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
          'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
          'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
          'New Jersey', 'New Mexico', 'New York', 'North Carolina',
          'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
          'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
          'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
          'Wisconsin', 'Wyoming']

pd.set_option("max_columns", None)
pd.set_option("max_colwidth", None)

df = pd.read_csv(input_original_data, header=None)

df1 = (df[(df[5] != 'Unassigned') &
          (df[5] != 'Federal Correctional Institution (FCI)') &
          (df[5] != 'Michigan Department of Corrections (MDOC)') &
          (df[5] != 'Out of MI') & (df[8] != str(0.0)) & (df[9] != str(0.0))]).reset_index(drop=True)

for i in range(len(states)):

    start_time = time.time()

    state = states[i]
    state_cases = []  # record cases for each state for all days summed over counties

    for j in range(days + 1):

        state_sum_cases = 0

        for k in range(len(df1)):  # sum cases of all counties for each day

            if state == df1.iloc[k, 6]:
                # since the actual infected from column 63 of df1
                # Also we want the daily cases from Mar 15
                # so we start from Mar 14 to include 1 diff from cumulative cases
                state_sum_cases += int(df1.iloc[k, j + 63])  # for infected cases
                # state_sum_cases += int(df1.iloc[k, j+64]) # for death cases

        state_cases.append(state_sum_cases)

    state_cases = np.diff(state_cases, n=1)  # to calculate daily infected/death cases from cumulative cases

    end_time = time.time()

    print("Time elapsed per state: ", end_time - start_time)

    with open(output_confirmed_inter, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(state_cases)
        csv_file.close()

    print(state, 'is done')

df2 = pd.read_csv(output_confirmed_inter, header=None)
data = df2.iloc[:, :].values
data = list(map(list, zip(*data)))
data = pd.DataFrame(data)

# X Matrix (Node Feature Matrix) saved as a csv file
data.to_csv(output_confirmed_cases_US, header=0, index=0)

os.remove(output_confirmed_inter)
############################################################################################################
############################################################################################################

# Generate the Adjacency Matrix (W) from Latitude and Longitude of 51 States

# confirmed covid-19 cumulative cases
output_W_inter = 'covid19_confirmed_W_matrix_inter.csv'

# Find the location of the states by averaging latitude and longitude of all counties/state
# location for each state
for i in range(len(states)):

    state = states[i]
    state_loc = []

    latitude = []
    longitude = []

    for j in range(len(df1)):  # average lat and long of all counties

        if state == df1.iloc[j, 6]:
            latitude.append(float(df1.iloc[j, 8]))
            longitude.append(float(df1.iloc[j, 9]))

    state_loc.append(np.mean(latitude))
    state_loc.append(np.mean(longitude))

    with open(output_W_inter, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(state_loc)
        csv_file.close()

    print(state, 'is done')

df2 = pd.read_csv(output_W_inter, header=None)

data = df2.iloc[:, :].values
data = list(map(list, zip(*data)))

output_W_inter_1 = 'covid19_confirmed_W_matrix_inter_1.csv'
output_W_final = 'covid19_confirmed_US_51_states_W_matrix_final.csv'

data = pd.DataFrame(data)
data.to_csv(output_W_inter_1, header=0, index=0)

df3 = pd.read_csv(output_W_inter_1, header=None)

vec = []
A = []
i = 0
j = 0

for i in range(len(states)):

    for j in range(df3.shape[0]):
        A.append(df3.iloc[j, i])

    vec.append(1)

    vec[i] = A

    A = []

distA = pdist(vec, metric='euclidean')
distB = squareform(distA)

for a in range(len(states)):

    A = []

    for b in range(len(states)):
        A.append(distB[a, b])

    with open(output_W_final, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(A)
        csv_file.close()

os.remove(output_W_inter)
os.remove(output_W_inter_1)

