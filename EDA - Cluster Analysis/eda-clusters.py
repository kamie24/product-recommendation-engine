#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis (EDA)
# 
# _Using Python for Data Exploration | Investments Case Study_
# 
# ***
# 
# ## Table of Contents
# 
# * [Introduction](#intro)
# * [General Outline](#outline)
# * [Import Required Libraries](#import)
# * [Load Data](#load)
# * [Explore Data - EDA](#eda)
#     * [Check Data](#check)
#     * [Explore Numerical Features by Cluster](#num-eda)
#         * [Create Histograms](#hist)
#         * [Create Box Plots](#boxplots)
#         * [Calculate Statistics](#stats)
#     * [Explore Categorical Features by Cluster](#cat-eda)
#         * [Create Bi-Variate Count Plots](#multi-countplots)
#         * [Create Violin Plots](#violin)
#         * [Create Swarm Plots](#multi-swarm)
# 
# ## Introduction <a id='intro'></a>
# 
# This EDA will try to reveal patterns, relationships, anomalies and outliers within the dataset. It will support early decision making by providing context around each of the historical products (investments) that were live on the company’s website at some point in time. Here are some leading questions to help us understand how we could possibly classify eventually these investments:
# 
# * Are there any correlations and distinctions amongst the dataset that make it strong enough to pursue a cluster analysis?
# * What type of variation occurs within the variables?
# * What type of covariation occurs between the variables?
# * Which values are the most common? Why?
# * Which values are rare? Why?
# * Can you see any unusual patterns?
# * What might explain them?
# * Are there subgroups in the data?
# * How are the observations within each cluster similar to each other?
# * How are the observations in separate clusters different from each other?
# * How can you explain or describe the clusters?
# * Why might the appearance of clusters be misleading (if possible)?
# * Are there any data points that don’t seem to fit the pattern, like outliers or anomalies?
# * Explore the distribution of the real estate KPIs (definitions from the glossary I provided). Is there anything unusual with the distribution of the variables?
# * Visualize the relationship between the variables (KPIs from the glossary) by showing the density with box plots.
# * Visualize the relationship between the categorical variables vs numerical ones. Do any patterns in the data exist?
# * Could any patterns be due to coincidence (i.e. random chance)?
# * How can you describe the relationship implied by any of the patterns?
# * How strong is the relationship implied by the pattern?
# * What other variables might affect the relationship?
# * Does the relationship change if you look at individual subgroups of the data?
# 
# ## General Outline <a id='outline'></a>
# 
# 1. Load data
# 2. Explore data - EDA
#     1. Univariate
#     2. Bivariate
#     3. Multi-variate



# Import the required libraries
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# Increase the width of cells
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:58% !important; }</style>"))


# [Go to the top](#top)
# 
# Read the dataset and load it into a Pandas dataframe.



pd.set_option('display.max_rows', 100)

# Read data.
df = pd.read_csv('k-means-clusters.csv')



# [Go to the top](#top)
# 
# 
# The Exploratory Data Analysis or EDA include the following steps:
# 
# * Review the available data and select specific variables of interest.
# * Check the quality of data.
# * Check for imbalances and create charts.
# * Identify opportunities, if any, to recode current variables or create new ones combining variables into a single measure.
# 
# The dataframe will be examined for the quality of the data. The types and shape of the data will be checked, as well as if there are any missing or duplicated records.


# Create a function to check the data.
def check_data(df): 
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()]).T.squeeze()
    duplicates = df.duplicated().sum()
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ratio = round((df.isnull().sum()/ obs) * 100, 2)
    skewness = df.skew()
    kurtosis = df.kurt() 
    print('Data shape: ', df.shape)
    print('Duplicates: ', duplicates)
    frame = {'types': types, 'counts': counts, 'uniques': uniques, 'nulls': nulls, 'distincts': distincts,
             'missing_ratio': missing_ratio, 'skewness': skewness, 'kurtosis': kurtosis}
    checks = pd.DataFrame(frame)
    display(checks)
    display(df.describe().T)

check_data(df)


# **Inference**
# 
# * We see a detailed information for each variable, such as descriptive statistics, number of missing values, distinct valus, etc.
# Plot histograms.



# Select numerical features
df_num = df.select_dtypes(include=['int64', 'float64'])

# Set style
sns.set_style("whitegrid")

# Plot histograms
count=1
plt.subplots(figsize=(18, 30))
for col in df_num.columns:
    plt.subplot(8, 3, count)
    sns.histplot(data=df, x=col, hue="Cluster",
                 fill=True, common_norm=False, palette="Set3",
                 alpha=.7, linewidth=0)
    count+=1
plt.tight_layout(pad=1.4)
plt.show()


# **Inference**
# 
# * We can observe separation for some variables - for example for "EM Last \$ Exposure @ Closing_x"
# 
# Box plots allow us to view the distribution of a parameter within bins. It's useful to see outliers too.


# Plot box plots
count=1
plt.subplots(figsize=(18, 30))
for col in df_num.columns:
    plt.subplot(5, 3, count)
    sns.boxplot(data=df, x='Cluster', y=col, palette='Set3')
    count+=1
plt.tight_layout(pad=1.4)
plt.show()


# **Inference**
# 
# * Now, we can observe better the distributions of each numerical feature by each cluster.
# * For instance, it seems that Cluster 0 has higher "capital_request" and "maximum_investment_amount".
# 
# Compute descriptive statistics by each cluster.



pd.set_option('display.max_rows', 500)

# Show descriptive stats for each cluster
stats = df.iloc[:, 1:].groupby(by=['Cluster']).describe(include='all').T
stats = stats.reset_index()
stats.rename(columns={"level_0": "Feature", "level_1": "Stats",
                      0: "Cluster_0", 1: "Cluster_1",
                      2: "Cluster_2", 3: "Cluster_3"}, inplace=True)

fig = go.Figure(data=[go.Table(
    header=dict(values=list(stats.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[stats.Feature, stats.Stats, stats.Cluster_0, stats.Cluster_1,
                       stats.Cluster_2, stats.Cluster_3],
               fill_color='lavender',
               align='left'))
])

fig.show()


# [Go to the top](#top)
# 
# ### Explore Categorical Features by Cluster
# #### Create Bi-Variate Count Plots
# 
# Show value counts by cluster.



# Select categorical columns
categorical = df.select_dtypes(include='object')
categorical.pop('id')

count=1
plt.subplots(figsize=(18, 55))
for col in categorical.columns:
    plt.subplot(8, 2, count)
    sns.countplot(y=col, hue='Cluster', palette="Set3", data=df,
                  order=df[col].value_counts().index, dodge=True)
    plt.legend(loc='lower right')
    count+=1
plt.tight_layout(pad=1.4)
plt.show()


# **Inference**
# 
# * We can easily see the most common values for each cluster.
# * For example, stabilized return on cost is observed only for cluster 1.
# 
# Draw categorical scatterplots with non-overlapping points.


for num in ['capital_request', 'maximum_investment_amount', 'minimum_investment_amount']:
    count=1
    plt.subplots(figsize=(18, 30))
    for col in ['entity_type', 'investment_strategy', 'subtype']:
        plt.subplot(3, 3, count)
        sns.violinplot(x=num, y=col, hue='Cluster', palette="Set3", data=df)
        count+=1
    plt.tight_layout(pad=1.4)
    plt.show()


# **Inference**
# 
# * We can see the distributions by each cluster.
# 
# Let's dive into more detail of the capital request, drawing swarm plots for two categorical variables.


plt.figure(figsize=(9, 5))
sns.swarmplot(x='entity_type', y='capital_request', hue='Cluster', palette="Set2", data=df, dodge=True);

