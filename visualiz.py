###############################################################################################
# Suppress matplotlib user warnings                                                           #
# Necessary for newer version of matplotlib                                                   #
# Credits: code adjusted from Udacity template                                                #
# https://github.com/udacity/machine-learning/blob/master/projects/finding_donors/visuals.py  #
###############################################################################################
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")

# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, precision_score


def distribution(data, transformed = False):
    """
    Visualization code for displaying skewed distributions of features
    """

    # Create figure
    fig = pl.figure(figsize = (11,5));

    # Skewed feature plotting 
    for i, feature in enumerate(['PREVAILING_WAGE']):
        ax = fig.add_subplot(1, 2, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s' Feature Distribution"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim((0, 1000000))
        ax.set_yticks([0, 200000, 400000, 600000, 800000, 1000000])
        ax.set_yticklabels([0, 200000, 400000, 600000, 800000, ">1000000"])

    # Plot aesthetics
    if transformed:
        fig.suptitle("Log-transformed Distributions of Continuous Financial Data Features", \
            fontsize = 16, y = 1.03)
    else:
        pass
        #fig.suptitle("Skewed Distributions of Continuous Data Features", \
            #fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()


def evaluate(results, metric, k):
    ind = 1
    width = 0.4       
    pl.xticks(range(0,1)) # Show no bar-labels
    pl.xlabel('Models')
    pl.ylabel('Scores')
 
    if k == 0: # Test & prediction times
        pl.xlabel('Models (left = prediction, right = training)')
        pl.ylabel('Time (in seconds)')
        for key, data_dict in results.items():
            x = data_dict.keys() 
            y = data_dict.values()
            if key == 'LogisticRegression':
                pl.bar(ind, y[k], color='#A00000', align='center', width = 0.3, label = 'Logit') # Prediction 
                pl.bar(ind+width, y[k+3], color='#A00000', align='center', width = 0.3, ) # Training
            elif key == 'RandomForestClassifier': 
                pl.bar(ind+width*2, y[k], color='#00A000', align='center', width = 0.3,  label = "RF") # Prediction 
                pl.bar(ind+width*3, y[k+3], color='#00A000', align='center', width = 0.3) # Training
            elif key == 'DecisionTreeClassifier': 
                pl.bar(ind+width*4, y[k], color='#00A0A0', align='center', width = 0.3, label = "DT") # Prediction 
                pl.bar(ind+width*5, y[k+3], color='#00A0A0', align='center', width = 0.3) # Training

            pl.suptitle("Prediction & Training Times", fontsize = 16, x = 0.53, y = .95)
            pl.legend(loc = 'upper left')
            
    elif k == 1: # Precision score
        for key, data_dict in results.items():
            if key == 'LogisticRegression':
                x = data_dict.keys() 
                y = data_dict.values()
                pl.bar(ind, y[k], color='#A00000', align='center', width = 0.2, label = 'Logit') # y[1] Precision - y[2] F-Score
            elif key == 'DecisionTreeClassifier': 
                x = data_dict.keys() 
                y = data_dict.values()
                pl.bar(ind+width, y[k], color='#00A0A0', align='center', width = 0.2, label = 'DT')  
            elif key == 'RandomForestClassifier': 
                x = data_dict.keys() 
                y = data_dict.values()
                pl.bar(ind+width*2, y[k], color='#00A000', align='center', width = 0.2, label = 'RF')  

        pl.axhline(y = metric, linewidth = 1, color = 'k', linestyle = 'dashed')
        pl.suptitle("Precision scores", fontsize = 16, x = 0.53, y = .95)
        pl.legend(loc = 'down left')
    
    elif k == 2: #F-score
        for key, data_dict in results.items():
            if key == 'LogisticRegression':
                x = data_dict.keys() 
                y = data_dict.values()
                pl.bar(ind, y[k], color='#A00000', align='center', width = 0.2, label = 'Logit') # y[1] Precision - y[2] F-Score
            elif key == 'DecisionTreeClassifier': 
                x = data_dict.keys() 
                y = data_dict.values()
                pl.bar(ind+width, y[k], color='#00A0A0', align='center', width = 0.2, label = 'DT')  
            elif key == 'RandomForestClassifier': 
                x = data_dict.keys() 
                y = data_dict.values()
                pl.bar(ind+width*2, y[k], color='#00A000', align='center', width = 0.2, label = 'RF') 

        pl.axhline(y = metric, linewidth = 1, color = 'k', linestyle = 'dashed')
        pl.suptitle("F-scores", fontsize = 16, x = 0.53, y = .95)
        pl.legend(loc = 'down left')

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (20,8))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.3, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  