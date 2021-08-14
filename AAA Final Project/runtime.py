# Garrick Morley
# SENG 309 / 2 PM
# Presenting on 8 / 11 / 2021
# Testing whether or not water is likely to be potable based on the .csv file that was read

# Import multible libraries for the code
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt  
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

# Import modules and set up abbreviations for them
import numpy as np
import pandas as pd
import seaborn as sns

# Import miltiple specific libraries for the code
from matplotlib.colors import ListedColormap
from scipy.stats import norm, boxcox
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
from scipy import stats
from tqdm import tqdm_notebook

# Import the Lucifer Machine Learning Framework
from luciferml.supervised.classification import Classification
from luciferml.preprocessing import Preprocess as prep

# Import the potentially necessary warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

dataset = pd.read_csv("water_potability.csv")

# Set the columns to the features that were included in the .csv file

cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Set the necessary feature values

features = dataset.iloc[:, :-1]

labels = dataset.iloc[:, -1]

# Label the accuracy scores

accuracy_scores =  {}

# TESTING


