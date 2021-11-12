import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
from IPython.display import SVG


    
def plot_speaker_feature_distribution(df, feature, n_bars = 10, figsize = (15, 4), **plot_kwargs): 
    """
    Function for plotting the distribution of the speaker feature with no weigths (each speaker in the Quotebank dataset
    counts for 1 irrespectively of the number of times he was quotes or the number of repetitions of each quote), with weigths
    corresponding to the quote counts and with the weigths corresponding to the total number of occurences of the quotes.
    The distribution is plotted as an historgram for continuous features (the only continuous feature is 'age') and as
    barplots (limiting to the n_bars tallest bars) if the feature is categorical.
    
    Params:
        df::[pd.DataFrame]
            DataFrame containing informations of each speaker, including the feature we are interested in.
        feature::[str]
            Name of the feature to be plotted.
        n_bars::[int]
            Only used when plotting categorical features. Limits the number of bars displayed in the barplot to the
            n_bars tallest ones.
        figsize::[tuple(float, float)]
            The size of the figure in which the 3 subplots will be drawn.
        plot_kwargs::[dict]
            Dictionary containing additional keyword arguments which will be passed to the plotting functions
            (sns.histplot for continuous features and sns.barplot for categorical features).
            
    Returns:
        None    
    """
    feature_titled = feature.replace('_', ' ').title()
    
    df = df[[feature, 'quote_count', 'num_occurrences']].rename(columns = {feature: feature_titled, 
                                                                           'quote_count': 'Quote Counts', 
                                                                           'num_occurrences': 'Number of Occurrences'})
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
            
            sns.barplot(x = top_n_most_common, y = [value_counts[key][weight_col] for key in top_n_most_common], 
                        ax = ax, color = color, **plot_kwargs)
            ax.set_ylabel('Counts')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
            ax.set_title("Not Weighted" if weight_col == 'Not Weighted' else "Weighted by " + weight_col)
        
    fig.suptitle(feature_titled + " Distribution")
    fig.tight_layout()
    
        

def plot_co_occurrence_matrix(data, feature_1, feature_2, figsize = (10, 10), keep_top_n = None, 
                              annot = True, fmt = '.0f', **heatmap_kwargs):
    """
    Function for plotting the co-occurrence matrix of features in data. Said features in data are expected to be columns of the 
    dataframe in which missing values are None, and otherwise a list of elements (categorical) is contained in the cell.
    The co-occurrency matrix is defined as the number of times a value in the list for feature_1 is observed together with a
    value in the list of feature_2 in the same line of the dataframe.
    The number of lines and columns displayed in the co-occurrency matrix can be limited with keep_top_n parameter (keeping only 
    most frequent combinations) to avoid overcrowding of xlabels or ylabels.
    If feature_1 and feature_2 are the same, the upper-right portion of the heatmap is hidden as it contains redundant values.
    
    Params:
        data::[pd.DataFrame]
            DataFrame containing informations of each speaker, including the features we are interested in.
        feature_1::[str]
            Name of the first feature to use in the co-occurrence matrix.
        feature_2::[str]
            Name of the second feature to use in the co-occurrence matrix.
        figsize::[tuple(float, float)]
            The size of the figure in which the 3 subplots will be drawn.
        keep_top_n::[int]
            If not None, limits the number of rows and columns shown in the co-occurrence matrix to this value (only keeping
            most frequent values).
        annot::[bool]
            Whether to write the number of occurrences in each cell of the displayed co-occurrence matrix.
            Passed directly to sns.heatmap.
        fmt::[str]
            The format with which to print the number of occurrences in each cell of the displayed co-occurrence matrix.
            Passed directly to sns.heatmap.
        heatmap_kwargs::[dict]
            Additional keyword arguments passed directly to sns.heatmap.
            
    Returns:
        None    
    """
    co_occurrence_counter = {}

    data = data[{feature_1, feature_2}].dropna(axis = 0, how = 'any')
    for _, row in data.iterrows():        
        values_feature_1 = set(row[feature_1])
        values_feature_2 = set(row[feature_2])
        
        for value_1 in values_feature_1:
            for value_2 in values_feature_2:                                
                co_occurrence_counter[(value_1, value_2)] = co_occurrence_counter.get((value_1, value_2), 0) + 1

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
    mask = np.triu(np.ones_like(df, dtype = bool)) if feature_1 == feature_2 else None
    
    sns.heatmap(df, square = True, mask = mask, annot = annot, fmt = fmt, norm = LogNorm(), **heatmap_kwargs)    
    
    plt.title(f"Heatmap of Quote Counts between {feature_1.title()} and {feature_2.title()}", pad = 20)    
    plt.ylabel(feature_1.title())
    plt.xlabel(feature_2.title())
    plt.show()
    
    
def plot_boxplot_for_each_discrete_value(data, feature_continuous, feature_discrete, figsize = (10, 10), keep_top_n = None, 
                                         filter_func = None, whis = float('inf'), **boxplot_kwargs):
    """
    Function for plotting boxplots of the distribution of a continuous for each value of a discrete feature. Said features in
    data are expected to be columns of the dataframe in which missing values are None, and otherwise a list of elements 
    for the discrete feature and a list of numbers, or a single number, for each cell in the continuous feature column.
    The number of different boxplots displayed can be limited with keep_top_n parameter (keeping only most frequent discrete values)
    to avoid overcrowding of xlabels.
    
    Params:
        data::[pd.DataFrame]
            DataFrame containing informations of each speaker, including the features we are interested in.
        feature_continuous::[str]
            Name of the continuous feature to plot the distribution of as boxplots.
        feature_discrete::[str]
            Name of the discrete feature for each value of which a boxplot will be generated.
        figsize::[tuple(float, float)]
            The size of the figure in which the boxplots will be drawn.
        keep_top_n::[int]
            If not None, limits the number of boxplots displayed (keeping only those corresponding to the most frequent discrete values).
        filter_func::[function]
            Function used for filtering values of the continuous feature. If not None, each value of continuous feature will be
            fed to the function and if it returns True, said value will be used in boxplot, otherwise it will be discarded.
        whis::[float]
            The maximum length of the whiskers in the boxplot.
            Passed directly to plt.boxplot.
        boxplot_kwargs::[dict]
            Additional keyword arguments passed directly to plt.boxplot.
            
    Returns:
        None    
    """
    distribution_per_discrete_value = {}

    data = data[{feature_continuous, feature_discrete}].dropna(axis = 0, how = 'any')
    for _, row in data.iterrows():  
        values_feature_discrete   = set(row[feature_discrete])
        values_feature_continuous = row[feature_continuous]
        
        # If not an iterable, make it iterable.
        if not isinstance(values_feature_continuous, (list, tuple, np.ndarray)):
            values_feature_continuous = [values_feature_continuous]
        
        # Apply filter for continuous values.
        if filter_func is not None:
            values_feature_continuous = [value for value in values_feature_continuous if filter_func(value)]
        
        for value_discrete in values_feature_discrete:
            tmp = distribution_per_discrete_value.get(value_discrete, [])
            tmp.extend(values_feature_continuous)
            distribution_per_discrete_value[value_discrete] = tmp
                  
    if keep_top_n is not None:
        n_measures_per_key = {k: len(v) for k, v in distribution_per_discrete_value.items()}
        most_common_discrete_values = sorted(n_measures_per_key, key = n_measures_per_key.get, reverse = True)[:keep_top_n]
        distribution_per_discrete_value = {k: distribution_per_discrete_value[k] for k in most_common_discrete_values}
           
    plt.figure(figsize = figsize)
    
    labels, data = [*zip(*distribution_per_discrete_value.items())]
    plt.boxplot(data, whis = whis, **boxplot_kwargs)
    plt.xticks(range(1, len(labels) + 1), labels, rotation = 90)
    plt.title(f"Distribution of {feature_continuous.title()} for each value of {feature_discrete.title()}", pad = 20)    
    plt.xlabel(feature_discrete.title())
    plt.ylabel(feature_continuous.title())
    plt.show()
    
    
def plot_hist_ecdf(data, x, weights = None, title = "", figsize = (16, 5), hist_kwargs = {}, ecdf_kwargs = {}):
    """
    Function for plotting histogram and empirical cumulative distribution function for column x in dataframe data.
    
    Params:
        data::[pd.DataFrame]
            DataFrame containing column x in which each row is a single scalar value.
        x::[str]
            Column in data for which to plot the distribution of.
        weights::[str | None]
            Column in data to use as weights of corresponding values in x. If None, each value has unitary weight.
        title::[str]
            Suptitle to histogram and  empirical cumulative distribution.
        figsize::[tuple(float, float)]
            The size of the figure in which the histogram and empirical cumulative distribution will be drawn.
        hist_kwargs::[dict]
            Additional keyword arguments to pass to sns.histplot.
        ecdf_kwargs::[dict]
            Additional keyword arguments to pass to sns.ecdfplot.
            
    Returns:
        None    
    """    
    fig, axes = plt.subplots(1, 2, figsize = figsize)
    
    sns.histplot(data, x = x, weights = weights, ax = axes[0], **hist_kwargs)
    sns.ecdfplot(data, x = x, weights = weights, ax = axes[1], **ecdf_kwargs)
    plt.suptitle(title)
    plt.show()
    
    
def plotly_to_svg(fig):
    """
    Function converting a plotly figure to an IPython SVG image.
    
    Params:
        fig::[plotly.graph_objects.Figure]
            Plotly figure to be converted to svg.
        
    Returns:
        svg_fig::[IPython.core.display.SVG]
            Input figure converted into IPython SVG image.
    """
    
    svg_fig_bytes = fig.to_image(format = "svg")
    return SVG(svg_fig_bytes)