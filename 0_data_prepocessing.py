# Graphs
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('EEG_Eye_State_Arff.csv')
# draw plots
sns.set(style='whitegrid', context='notebook')
cols = ['AF3', 'F7', 'F3', 'FC5', 'T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
sns.pairplot(dataset[cols], size=1)
plt.show()


#Correlation Analysis

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('EEG_Eye_State_Arff.csv')
#Correlation matrix
corr_matrix=dataset.corr(method='pearson')
print(corr_matrix)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('EEG_Eye_State_Arff.csv')

ax=sns.boxplot(x=dataset['AF3'])




