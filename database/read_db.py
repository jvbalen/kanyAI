# This python file reads the sample database

import numpy as np

f = np.genfromtxt("sample_database.csv", delimiter=";", dtype=object)

trackids = f[:,0]
artists = f[:,1]
titles = f[:,2]
years = f[:,3]
genres = f[:,4]
