from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.display import SVG

from src.utils_sparse_matrix import get_distribution_per_discrete_value


    
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
    
    
def plot_hist(data, color, bins, xlog = False, ylog = False, **kwargs):
    """
    Function allowing to plot an histogram of the provided data as well as a visualization of the mean and std
    of the data on top of the same histogram.
    
    Params:
        data::[iterable]
            Data we wish to plot the histogram of.
        color::[tuple]
            Color to use for the histogram.
        bins::[int | iterable]
            If int, number of bins to use in the histogram. If iterable, numbers representing the breaks in consecutive bins.
        xlog::[bool]
            Boolean describing whether the x-axis should be in log scale.
        ylog::[bool]
            Boolean describing whether the x-axis should be in log scale.
        kwargs::[dict]
            Any additional arguments which should be passed to sns.histplot call.
            
    Returns:
        bins::[np.array]
            Bins used in the histogram in the plot.
    """
    # Compute hist bins if necessary.
    if isinstance(bins, int):
        if xlog:
            bins = np.linspace(np.log10(data.min()), np.log10(data.max()), bins + 1)
        else:
            bins = np.linspace(data.min(), data.max(), bins + 1)
    
    # Draw histogram.
    sns.histplot(data, bins = bins, color = color, log_scale = (xlog, ylog), **kwargs)
    
    return bins


def plot_alphas_vs_impurity_tree(alphas, impurities):
    """
    Function plotting the decision tree's impurities vs post-pruning alphas provided as parameters.
    
    Params:
        alphas::[np.array]
            Array of post-pruning alphas with which the tree was pruned and for which the pruned tree's impurity was computed.  
        impurities::[np.array]
            Array of computed pruned tree's impurity for each value in alphas paramater.
    
    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize = (15, 5))
    for ax in axes:
        ax.plot(alphas, impurities, marker = " ", drawstyle = "steps-post")
        ax.set_xlabel("Post-pruning alpha")
        ax.set_ylabel("Total impurity of leaves")

    axes[1].set_xscale('log')   
    fig.suptitle('Total impurity of leaves vs post-pruning alpha for training set')
    plt.show()


def plot_boxplot_for_each_sparse_feature(features, num_occurrences, features_cols_titles, features_prefix, 
                                         figsize = (15, 6), keep_top_n = None, log = False, 
                                         whis = float('inf'), **boxplot_kwargs):
    """
    Function plotting distribution of parameter num_occurrences for each column in features parameter whose title
    starts with features_prefix. The distribution is plotted as boxplots arranged horizontally.
        
    Params:
        features::[np.array | scipy.sparse.matrix]



    
    Returns:
        None
    """
    
    distribution_per_discrete_value = get_distribution_per_discrete_value(features, num_occurrences,
                                                                          features_cols_titles, features_prefix,
                                                                          keep_top_n)
    
    name_feature_discrete = features_prefix.replace('_', ' ').title()
    name_feature_continuous = 'Number of Occurrences'

    plt.figure(figsize = figsize)
    
    labels, data = [*zip(*distribution_per_discrete_value.items())]
    plt.boxplot(data, whis = whis, **boxplot_kwargs)
    plt.xticks(range(1, len(labels) + 1), labels, rotation = 90)    
    plt.title(f"Distribution of {name_feature_continuous} for each value of {name_feature_discrete}", pad = 20)    
    plt.xlabel(name_feature_discrete)
    plt.ylabel(name_feature_continuous)
    
    if log:
        plt.yscale('log')
    
    plt.show()
    
    
def plot_non_overlapped_hist(features, num_occurrences, features_cols_titles, features_prefix, 
                             bins = 250, n_subplots_per_line = 6, val_log = False, count_log = False,
                             keep_top_n = None):
    
    distribution_per_discrete_value = get_distribution_per_discrete_value(features, num_occurrences,
                                                                          features_cols_titles, features_prefix,
                                                                          keep_top_n)
    
    name_feature_discrete = features_prefix.replace('_', ' ').title()
    name_feature_continuous = 'Number of Occurrences'
    
    n_lines = int(np.ceil(len(distribution_per_discrete_value) / n_subplots_per_line))
    fig, axes = plt.subplots(n_lines, n_subplots_per_line, figsize = (15, 6 * n_lines))
    axes = axes.ravel()
    
    val_min = min(values.min() for values in distribution_per_discrete_value.values())
    val_max = max(values.max() for values in distribution_per_discrete_value.values())
    
    if val_log:
        bins = np.linspace(np.log10(val_min), np.log10(val_max), bins + 1)
    else:
        bins = np.linspace(val_min, val_max, bins + 1)
        
    palette = cycle(iter(sns.color_palette()))
    for i, ((feature_value, distrib), ax) in enumerate(zip(distribution_per_discrete_value.items(), axes)):

        # Draw histogram.
        sns.histplot(y = distrib, bins = bins, log_scale = (count_log, val_log), color = next(palette), element = "step", ax = ax)
        
        ax.set_ylim(bottom = val_min, top = val_max)

        ax.set_title(feature_value, pad = 30)
                
        ax.set_ylabel('' if i % n_subplots_per_line else name_feature_continuous)
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
         
    for ax in axes[i+1:]:
        ax.set_axis_off()
        
    fig.tight_layout()
    
    
    
def plot_boxplots_scores_results(results, n_subplots_per_line = 5, limit_between_0_and_1 = False, scores_to_not_show = []):
    df = pd.DataFrame(results)

    train_scores_df = df['train_scores'].apply(pd.Series)
    val_scores_df = df['val_scores'].apply(pd.Series)
    
    scores_to_show = [score for score in train_scores_df.columns if score not in scores_to_not_show]
    
    n_lines = int(np.ceil(len(scores_to_show) / n_subplots_per_line))
    fig, axes = plt.subplots(n_lines, n_subplots_per_line, figsize = (15, 5 * n_lines))
    axes = axes.ravel()
    
    for i, (score, ax) in enumerate(zip(scores_to_show, axes)):
        ax.boxplot([train_scores_df[score], val_scores_df[score]], labels = ["Training Set", "Validation Set"], whis = 1e6)
        ax.scatter([1] * len(train_scores_df[score]), train_scores_df[score], marker = '.')
        ax.scatter([2] * len(val_scores_df  [score]), val_scores_df  [score], marker = '.')
        ax.set_title(score.replace('_', ' ').title(), pad = 30)
                
        if not i % n_subplots_per_line:
            ax.set_ylabel('Score value')
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if limit_between_0_and_1:
            ax.set_ylim(top = 1, bottom = 0)
         
    for ax in axes[i+1:]:
        ax.set_axis_off()
        
    fig.tight_layout()
    
    
def plot_boxplots_coefs(results, features_cols_titles, figsize = (15, 300)):
    plt.figure(figsize = figsize)
    df = pd.DataFrame(results)

    coefs_df = pd.DataFrame(np.vstack(df['coefs'].values), columns = features_cols_titles)
    
    pvalues_df = pd.DataFrame(np.vstack(df['pvalues'].values), columns = features_cols_titles)
    
    # Combine pvalues across runs by Fisher's method. Ignore warning due to pvalues equal to 0.
    with np.errstate(divide = 'ignore'):
        combined_pvalues = pvalues_df.apply(lambda col: scipy.stats.combine_pvalues(col, method='fisher')[1], axis = 0)
    
    combined_pvalues = combined_pvalues.sort_values(ascending = False)
    
    renamed_cols = {name: f"{name} (combined pvalue: {pvalue:.5f})" for name, pvalue in combined_pvalues.iteritems()}
    coefs_df = coefs_df[combined_pvalues.index].rename(columns = renamed_cols)
    
    plt.boxplot(coefs_df, labels = coefs_df.columns, vert = False, whis = 3)
    plt.grid()
    
    
    
    
    
def plot_tree_results(results, features_cols_titles, n_subplots_per_line = 3, scores_to_not_show = []):
    
    # Reshaping list of dict of dict into a useable dataframe.
    df = pd.concat([pd.DataFrame.from_dict(x, orient='index') for x in results])
    df.index.name = 'pruning_alpha'
    df = df.reset_index()
    

    train_scores_df = df['train_scores'].apply(pd.Series)
    for col in train_scores_df:
        df[('train', col)] = train_scores_df[col]

    val_scores_df = df['val_scores'].apply(pd.Series)
    for col in train_scores_df:
        df[('val', col)] = val_scores_df[col]

    df = df.drop(columns = ['train_scores', 'val_scores'])
    scores = train_scores_df.columns.tolist()
    scores = [score for score in scores if score not in scores_to_not_show]

    n_lines = int(np.ceil(len(scores) / n_subplots_per_line))
    fig, axes = plt.subplots(n_lines, n_subplots_per_line, figsize = (15, 4 * n_lines))
    axes = axes.ravel()
    
    for i, (score, ax) in enumerate(zip(scores, axes)):

        scores = df[['pruning_alpha', ('train', score), ('val', score)]]        
        
        scores = scores.groupby('pruning_alpha').agg(['mean', 'max', 'min'])

        ax.plot(scores.index, scores[('train', score, 'mean')], label = 'Training Set')
        ax.fill_between(scores.index, scores[('train', score, 'min')], scores[('train', score, 'max')], alpha = 0.3)

        ax.plot(scores.index, scores[('val', score, 'mean')], label = 'Validation Set')
        ax.fill_between(scores.index, scores[('val', score, 'min')], scores[('val', score, 'max')], alpha = 0.3)

        ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel('Post-Pruning Alpha')
        ax.set_title(score.replace('_', ' ').title())
        
        if not i % n_subplots_per_line:
            ax.set_ylabel('Score value')
            
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
    for ax in axes[i+1:]:
        ax.set_axis_off()    
       
    fig.suptitle('Training and Validation scores for different Post-Pruning Alphas', y = 1)
    fig.tight_layout(h_pad = 4)