import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np

    
def plot_speaker_feature_distribution(df, feature, n_bars = 10, figsize = (15, 4), **plot_kwargs): 
    """ Function for plotting the distribution of the speaker features with no weigths,  with weigths corresponding to the quote counts and with the weigths corresponding to the number of occurences of the quotes.
    
    Params:
        df::[DataFrame]
            DataFrame containing informations of each speaker.  
        feature::[str]
            Name of the feature that is plotted.
        n_bars::[int]
            Number 
    
    """
    
    feature_titled = feature.replace('_', ' ').title()
    
    df = df[[feature, 'quote_count', 'num_occurrences']].rename(columns = {feature: feature_titled, 'quote_count': 'Quote Counts', 'num_occurrences': 'Number of Occurrences'})
    df.dropna(axis = 0, how = 'any', inplace = True)
    df.reset_index(drop = True, inplace = True)
        
    fig, axes = plt.subplots(1, 3, figsize = figsize)
    
    if feature_titled == 'Age':
        
        for ax, weight_col in zip(axes, (None, 'Quote Counts', 'Number of Occurrences')):
            sns.histplot(data = df, x = feature_titled, weights = weight_col, ax = ax, **plot_kwargs)
            ax.set_title("Not Weighted" if weight_col is None else "Weighted by " + weight_col)                
    else:
        value_counts = {}
        for _, row in df.iterrows():
            for value in row[feature_titled]:
                if value not in value_counts:
                    value_counts[value] = {'Not Weighted': 0, 'Quote Counts': 0, 'Number of Occurrences': 0}

                value_counts[value]['Not Weighted'] += 1
                value_counts[value]['Quote Counts'] += row['Quote Counts']
                value_counts[value]['Number of Occurrences'] += row['Number of Occurrences']
                
        for ax, weight_col in zip(axes, ('Not Weighted', 'Quote Counts', 'Number of Occurrences')):
            top_n_most_common = sorted(value_counts, key = lambda key: value_counts[key][weight_col], reverse = True)[:n_bars]
            
            color = sns.color_palette()[0]
            
            sns.barplot(x = top_n_most_common, y = [value_counts[key][weight_col] for key in top_n_most_common], ax = ax, color = color, **plot_kwargs)
            ax.set_ylabel('Counts')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
            ax.set_title("Not Weighted" if weight_col == 'Not Weighted' else "Weighted by " + weight_col)
        
    fig.suptitle(feature_titled + " Distribution")
    fig.tight_layout()
    
    
    

def plot_co_occurrence_matrix(data, feature_1, feature_2, figsize = (10, 10), weight = None, keep_top_n = None, 
                              annot = True, fmt = '.0f', **heatmap_kwargs):
    
    co_occurrence_counter = {}

    data = data[{feature_1, feature_2}].dropna(axis = 0, how = 'any')
    for _, row in data.iterrows():        
        values_feature_1 = set(row[feature_1])
        values_feature_2 = set(row[feature_2])
        
        for value_1 in values_feature_1:
            for value_2 in values_feature_2:                
                weight_to_add = 1 if weight is None else row[weight]
                
                co_occurrence_counter[(value_1, value_2)] = co_occurrence_counter.get((value_1, value_2), 0) + weight_to_add

    df = pd.Series(co_occurrence_counter.values(), 
                   index = pd.MultiIndex.from_tuples(co_occurrence_counter.keys()), 
                   dtype = 'int').unstack()
        
    if keep_top_n is not None:
        marginals_rows = df.sum(axis = 1)
        most_common_incices = marginals_rows.sort_values(ascending = False)[:keep_top_n].index
        marginals_cols = df.sum(axis = 0)
        most_common_cols = marginals_cols.sort_values(ascending = False)[:keep_top_n].index        
        df = df[df.index.isin(most_common_incices)][most_common_cols]

    # Sort columns and lines in alphabetical order to correctly display heatmap. 
    df = df[sorted(df.columns)]
    df = df.sort_index()
    
    plt.figure(figsize = figsize)
    
    # If the co-occurrence matrix is for the same feature, we can hide the upper part of the matrix as it is redundant. 
    if feature_1 == feature_2:
        mask = np.triu(np.ones_like(df, dtype = bool))
    else:
        mask = None
    
    sns.heatmap(df, square = True, mask = mask, annot = annot, fmt = fmt, norm = LogNorm(), **heatmap_kwargs)    
    
    plt.title(f"Heatmap of Quote Counts between {feature_1.title()} and {feature_2.title()}", pad = 20)    
    plt.xlabel(feature_1.title())
    plt.ylabel(feature_2.title())
    plt.show()