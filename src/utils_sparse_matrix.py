def get_columns(features, features_cols_titles, columns):
    """
    Utilitary function returning the columns of features parameter which have title in columns parameter.
    The titles of all columns is provided by the features_cols_titles parameter.
        
    Params:
        features::[np.array | scipy.sparse.matrix]
            Matrix from which we want to extract a subset of columns.
        features_cols_titles::[list | tuple]
            List of strings containing the name assigned to each column in features parameter.
        columns::[list | tuple | str]
            List of strings (or single string) containing the subset of columns we want to extract from the matrix.
            
    Returns:
        feature_columns::[np.array | scipy.sparse.matrix]
            Desired columns extracted from features matrix.
    """
    if isinstance(columns, str):
        columns = [columns]
        
    return features[:, [features_cols_titles.index(col) for col in columns]]


def split_num_occurrences_on_one_column(features, num_occurrences, features_cols_titles, binary_feature):
    """
    Utilitary function returning the distribution of num_occurrences parameter when the value of column binary_feature in features
    is 1 or 0. The titles of all columns is provided by the features_cols_titles parameter.
        
    Params:
        features::[np.array | scipy.sparse.matrix]
            Matrix from which we want to extract the column which we will then use to split the distributions with.
        num_occurrences::[np.array]
            Array to be splitted into two separate distributions.
        features_cols_titles::[list | tuple]
            List containing the name assigned to each column in features parameter.
        binary_feature::[str]
            The name of the column in features which should be used to choose which elements of num_occurrences belong to which
            distribution.
            
    Returns:
        split_num_occurrences::[tuple]
            Tuple containing the distributions of num_occurrences when binary feature takes value 0 as first element
            and that when binary feature takes value 1 as second element.
    """
    # Get index where feature of interest.
    col_values = get_columns(features, features_cols_titles, binary_feature).toarray().ravel()

    split_num_occurrences = (num_occurrences[col_values == 0], num_occurrences[col_values == 1])
    
    return split_num_occurrences


def get_distribution_per_discrete_value(features, num_occurrences, features_cols_titles, features_prefix,
                                        keep_top_n = None, return_also_reciprocal_distribution = False):
    """
    Utilitary function returning the distribution of num_occurrences parameter for each possible value of features
    whose title starts with features_prefix (each column of features encodes a binary feature, and this fonction returns
    the elements of num_occurrences for which this feature takes value 1). Optionally, the distribution of elements when 
    a column takes value 0 can also be returned, and the columns corresponding to the least common features can be dropped.
    
    Params:
        features::[np.array | scipy.sparse.matrix]
            Matrix from which we want to extract the column which we will then use to split the distributions with.
        num_occurrences::[np.array]
            Array to be splitted into two separate distributions.
        features_cols_titles::[list | tuple]
            List containing the name assigned to each column in features parameter.
        features_prefix::[str]
            The prefix of the names of columns we want to extract from features matrix.
        keep_top_n::[int | None]
            Number of most-frequently occurring features (columns for which the elements are 1) to keep in returned value.
        return_also_reciprocal_distribution::[bool]
            Wether to return both the elements of num_occurrences for which the corresponding column in features has value 1,
            or also those for which the value is 0 (collecting them into a 2-elements tuple).
            
    Returns:
        distribution_per_discrete_value::[dict]
            Dictionary with as keys the possible values that features starting by features_prefix can take, and as values
            either an array with the distribution of num_occurrences when that feature is 1 (if return_also_reciprocal_distribution
            is False) or a tuple with both the distribution when the feature is 1 and 0 (if return_also_reciprocal_distribution
            is True).
    """
    cols_to_keep = [col for col in features_cols_titles if col.startswith(features_prefix)]
    features = get_columns(features, features_cols_titles, cols_to_keep)
    
    features_cols_titles = [col.replace(features_prefix, '').replace('_', ' ').strip().title() for col in cols_to_keep]
    
    distribution_per_discrete_value = {}
    
    for col_idx in range(len(features_cols_titles)):
        col_values = features[:, col_idx].toarray().ravel()
        col_name   = features_cols_titles[col_idx]
        
        if return_also_reciprocal_distribution:
            distribution_per_discrete_value[col_name] = (num_occurrences[col_values == 1], num_occurrences[col_values == 0])
        else:
            distribution_per_discrete_value[col_name] = num_occurrences[col_values == 1]
      
    if keep_top_n is not None:
        n_measures_per_key = {k: len(v) for k, v in distribution_per_discrete_value.items()}
        most_common_discrete_values = sorted(n_measures_per_key, key = n_measures_per_key.get, reverse = True)[:keep_top_n]
        distribution_per_discrete_value = {k: distribution_per_discrete_value[k] for k in most_common_discrete_values}
        
    return distribution_per_discrete_value