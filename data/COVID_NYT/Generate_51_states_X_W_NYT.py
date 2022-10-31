# Generate the Node Feature and Adjacency Matrices (csv files) for NYT time-series dataset for US states
import pandas as pd
import numpy as np
import csv
import os
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# Generate the V or X Matrix (Node Feature Matrix) from NYT COVID time-series
'''
Extract the time series of the total number of confirmed cases/deaths in each state in the original data set.
parameters settings
days: the number of days collected
'''
days = 623  # Mar 18,2020 - Nov 30,2021

# confirmed covid-19 cumulative cases/deaths
input_original_data = 'us-states.csv'

# output_confirmed_inter = 'covid19_NYT_confirmed_US_51_states_inter.csv'
output_confirmed_inter = 'covid19_NYT_deaths_US_51_states_inter.csv'

# output_confirmed_cases_US = 'covid19_NYT_confirmed_US_51_states_X_matrix_final.csv'
output_confirmed_cases_US = 'covid19_NYT_deaths_US_51_states_X_matrix_final.csv'

pd.set_option("max_columns", None)
pd.set_option("max_colwidth", None)

df1 = pd.read_csv(input_original_data)

# We included Washington D.C. as a node for the graph similar to the states of US
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

df2 = (df1[(df1['date'] >= '2020-03-17') &
           (df1['date'] <= '2021-11-30')]).reset_index(drop=True)

statelist = sorted(list(set(df2['state']).intersection(set(states))))
datelist = sorted(list(set(df2['date'])))

for i in range(len(statelist)):

    start_time = time.time()

    state = statelist[i]

    state_cases = []  # record cases for each state

    for j in range(len(datelist)):
        df3 = df2[df2['state'] == state].reset_index(drop=True)

        # state_cases.append(df3['cases'][j])
        state_cases.append(df3['deaths'][j])

    state_cases = np.diff(state_cases, n=1)  # to calculate daily infected cases from cumulative cases

    end_time = time.time()

    print("Time elapsed per state: ", end_time - start_time)

    with open(output_confirmed_inter, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(state_cases)
        csv_file.close()

    print(state, 'is done')

df4 = pd.read_csv(output_confirmed_inter, header=None)
data = df4.iloc[:, :].values
data = list(map(list, zip(*data)))
data = pd.DataFrame(data)

# X Matrix (Node Feature Matrix) saved as a csv file
data.to_csv(output_confirmed_cases_US, header=0, index=0)

os.remove(output_confirmed_inter)

# We used the same Adjacency Matrix (W) for US states as generated from JHU Dataset

