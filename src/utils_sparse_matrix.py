def get_columns(features, features_cols_titles, columns):
    if isinstance(columns, str):
        columns = [columns]
        
    return features[:, [features_cols_titles.index(col) for col in columns]]


def split_num_occurrences_on_one_column(features, num_occurrences, features_cols_titles, binary_feature):
    """
    """
    # Get index where feature of interest.
    col_values = get_columns(features, features_cols_titles, binary_feature).toarray().ravel()

    split_num_occurrences = [num_occurrences[col_values == 0], num_occurrences[col_values == 1]]
    
    return split_num_occurrences


def get_distribution_per_discrete_value(features, num_occurrences, features_cols_titles, features_prefix,
                                        keep_top_n = None, return_also_reciprocal_distribution = False):
    cols_to_keep = [col for col in features_cols_titles if col.startswith(features_prefix)]
    features = get_columns(features, features_cols_titles, cols_to_keep)
    
    features_cols_titles = [col.replace(features_prefix, '').replace('_', '').title() for col in cols_to_keep]
    
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