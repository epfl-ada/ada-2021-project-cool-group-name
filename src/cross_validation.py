import time
import numpy as np
import scipy.stats
import scipy.sparse
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import *

import src.utils as utils






    
def _folds_generator(X, y, cv_n_splits, problem_type, verbose, train_on_single_fold = False):
    stratified_k_folds = StratifiedKFold(n_splits = cv_n_splits, shuffle = True)
    
    if problem_type == 'regression':
        folds_idx = stratified_k_folds.split(np.zeros_like(y), utils.get_percentile_stratifier(y))
    
    elif problem_type == 'classification':
        folds_idx = stratified_k_folds.split(np.zeros_like(y), y)

    else:
        raise ValueError("Parameter 'problem_type' must be either 'regression' or 'classification'.")
    
    # Generator to list.
    folds_idx = list(folds_idx)
    
    if train_on_single_fold:
        folds_idx = [elem[::-1] for elem in folds_idx]
    
    for i, (train_idx, val_idx) in enumerate(folds_idx):
        if verbose:
            print(f"Starting fold {i + 1} of {len(folds_idx)}.")
            print(f"Fold contains {len(train_idx)} training samples and {len(val_idx)} validation samples.")
            start = time.time()
        
        yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]
        
        if verbose:
            print(f"Finished Fold {i + 1} of {len(folds_idx)} in {int(time.time() - start)} seconds.")


def _scores_regression(y, preds):
    return {'r2'                   : r2_score(y, preds),
            'mse'                  : mean_squared_error(y, preds),
            'explained_variance'   : explained_variance_score(y, preds),
            'max_error'            : max_error(y, preds),
            'median_absolute_error': median_absolute_error(y, preds)}           


def _scores_classification(y, preds):
    return {'accuracy'  : accuracy_score(y, preds),
            'precision' : precision_score(y, preds),
            'recall'    : recall_score(y, preds),
            'f1'        : f1_score(y, preds)}


@utils.cache_to_file_pickle("cross_validation-baseline_regression_cv", ignore_kwargs = ['features', 'verbose'])
def baseline_regression_cv(features, num_occurrences, features_cols_titles, cv_n_splits, verbose = False):
    
    num_occurrences = np.array(num_occurrences)
        
    print("Training on {} samples with {} features".format(*features.shape))
        
    results = []
    
    folds_generator = _folds_generator(features, num_occurrences, cv_n_splits, 'regression', verbose)
    for features_train, features_val, num_occurrences_train, num_occurrences_val in folds_generator:
        
        train_mean = num_occurrences_train.mean()
        
        current_fold_results = {}
        
        for key, X, y in [('train_scores', features_train, num_occurrences_train),
                          ('val_scores'  , features_val  , num_occurrences_val)]:
            
            preds = np.full_like(y, train_mean)
            
            current_fold_results[key] = _scores_regression(y, preds)
            
        results.append(current_fold_results)
            
    return results





@utils.cache_to_file_pickle("cross_validation-linear_regression_cv", ignore_kwargs = ['features', 'verbose'])
def linear_regression_cv(features, num_occurrences, features_cols_titles, cv_n_splits, verbose = False):

    def add_intercept(features, features_cols_titles):
        n_samples, n_features = features.shape

        features = scipy.sparse.hstack([np.ones((n_samples, 1), dtype = 'uint8'), features]).tocsr()
        features_cols_titles = ['intercept'] + features_cols_titles

        return features, features_cols_titles
    
    
    def compute_pvalues(X, y, coefs):
        preds = X.dot(coefs)
        
        df = len(preds) - len(coefs)
        
        adjusted_mse = np.sum((y - preds)**2) / df
        
        var_coefs = adjusted_mse * scipy.sparse.linalg.inv(X.T.dot(X)).diagonal()
                
        # Compute t-statistics, avoiding warnings due to numerical errors.
        t_statistic_coefs = [coef / np.sqrt(var) if var > 0 else float('inf') for coef, var in zip(coefs, var_coefs)]

        p_values = [2 * (1 - scipy.stats.t.cdf(np.abs(t_statistic), df)) for t_statistic in t_statistic_coefs]
        
        return p_values
    
    num_occurrences = np.array(num_occurrences)
    
    features, features_cols_titles = add_intercept(features, features_cols_titles) 
    
    print("Training on {} samples with {} features".format(*features.shape))
        
    results = []
    
    folds_generator = _folds_generator(features, num_occurrences, cv_n_splits, 'regression', verbose)
    for features_train, features_val, num_occurrences_train, num_occurrences_val in folds_generator:
        
        coefs = scipy.sparse.linalg.lsqr(features_train, num_occurrences_train)[0]          
                
        current_fold_results = {'coefs': coefs, 
                                'pvalues': compute_pvalues(features_train, num_occurrences_train, coefs)}
        
        for key, X, y in [('train_scores', features_train, num_occurrences_train),
                          ('val_scores'  , features_val  , num_occurrences_val)]:
            
            preds = X.dot(coefs)
            
            current_fold_results[key] = _scores_regression(y, preds)
            
            # For linear regression, add adjusted R2 score to allow comparison between models with less and less features.
            r2 = current_fold_results[key]['r2']
            adjusted_r2 = 1 - (1 - r2) * (len(num_occurrences_train) - 1) / (len(num_occurrences_train) - len(coefs) - 1)
            current_fold_results[key]['adjusted_r2'] = adjusted_r2
            
        results.append(current_fold_results)
            
    return results




@utils.cache_to_file_pickle("cross_validation-tree_regression_cv", ignore_kwargs = ['features', 'verbose'])
def tree_regression_cv(features, num_occurrences, post_pruning_alphas, max_depth, cv_n_splits, verbose = False):
    
    num_occurrences = np.array(num_occurrences)
    
    print("Training on {} samples with {} features".format(*features.shape))
    
    results = []
    
    # Train on single fold, evaluate on all other folds for computational cost reasons.
    folds_generator = _folds_generator(features, num_occurrences, cv_n_splits, 'regression', verbose, 
                                       train_on_single_fold = True)
    for features_train, features_val, num_occurrences_train, num_occurrences_val in folds_generator:
        
        # Train unpruned model a single time, with a specified maximum depth but otherwise no pre-pruning.
        # At each branch, tree will always be allowed to choose best split. As such, it should be deterministic
        # when provided the same data, irrespectively of post-pruning parameter ccp_alpha.
        unpruned_model = DecisionTreeRegressor(max_depth = max_depth).fit(features_train, num_occurrences_train) 
        
        current_fold_results = {alpha: {} for alpha in post_pruning_alphas}
        
        for alpha in post_pruning_alphas:
            # Instead of retraining multiple times the same tree and then post-pruning each time, train a single time,
            # deepcopy the instance and prune it once per each post-pruning parameter and compute scores. 
            model = deepcopy(unpruned_model)
            model.set_params(ccp_alpha = alpha)
            model._prune_tree()
            
            for key, X, y in [('train_scores', features_train, num_occurrences_train),
                              ('val_scores'  , features_val  , num_occurrences_val)]:
            
                preds = model.predict(X)

                current_fold_results[alpha][key] = _scores_regression(y, preds)
            
        results.append(current_fold_results)
            
    return results


@utils.cache_to_file_pickle("cross_validation-baseline_classification_cv", ignore_kwargs = ['features', 'verbose'])
def baseline_classification_cv(features, labels, cv_n_splits, verbose = False):
    
    labels = np.array(labels)
        
    print("Training on {} samples with {} features".format(*features.shape))
        
    strategies = ['uniform', 'stratified', 'constant_0', 'constant_1']
    results = {strategy: [] for strategy in strategies}
    
    folds_generator = _folds_generator(features, labels, cv_n_splits, 'classification', verbose)
    for features_train, features_val, labels_train, labels_val in folds_generator:
                
        for strategy in strategies:
            current_fold_strategy_results = {}
            
            if strategy.startswith('constant'):
                model = DummyClassifier('constant', constant = int(strategy.split('_')[-1]))
            else:
                model = DummyClassifier(strategy)

            model.fit(features_train, labels_train)

            for key, X, y in [('train_scores', features_train, labels_train),
                              ('val_scores'  , features_val  , labels_val)]:

                preds = model.predict(X)

                current_fold_strategy_results[key] = _scores_classification(y, preds)

            results[strategy].append(current_fold_strategy_results)
            
    return results




@utils.cache_to_file_pickle("cross_validation-linear_svm_classification_cv", ignore_kwargs = ['features', 'verbose'])
def linear_svm_classification_cv(features, labels, balanced_class_weight, cv_n_splits, verbose = False):    
    
    labels = np.array(labels)
    
    print("Training on {} samples with {} features".format(*features.shape))
    print(f"Positive samples: {np.sum(labels == 1)}, negative samples: {np.sum(labels == 0)}")

    results = []
    
    folds_generator = _folds_generator(features, labels, cv_n_splits, 'classification', verbose)
    for features_train, features_val, labels_train, labels_val in folds_generator:
        
        model = LinearSVC(dual = False, max_iter = 10000, class_weight = 'balanced' if balanced_class_weight else None)
        model.fit(features_train, labels_train) 
                
        current_fold_results = {}
        
        for key, X, y in [('train_scores', features_train, labels_train),
                          ('val_scores'  , features_val  , labels_val)]:
            
            preds = model.predict(X)
            
            current_fold_results[key] = _scores_classification(y, preds)
            
            # For support vector machines, add sparsity to allow comparison between models with less and less features.
            current_fold_results[key]['sparsity'] = np.mean(np.abs(model.coef_) < 1e-8)
            
        results.append(current_fold_results)
            
    return results


@utils.cache_to_file_pickle("cross_validation-tree_classification_cv", ignore_kwargs = ['features', 'verbose'])
def tree_classification_cv(features, labels, post_pruning_alphas, max_depth, cv_n_splits, verbose = False):
    
    labels = np.array(labels)
    
    print("Training on {} samples with {} features".format(*features.shape))
    print(f"Positive samples: {np.sum(labels == 1)}, negative samples: {np.sum(labels == 0)}")
        
    results = []
    
    # Train on single fold, evaluate on all other folds for computational cost reasons.
    folds_generator = _folds_generator(features, labels, cv_n_splits, 'classification', verbose, 
                                       train_on_single_fold = True)
    for features_train, features_val, labels_train, labels_val in folds_generator:
        
        # Train unpruned model a single time, with a specified maximum depth but otherwise no pre-pruning.
        # At each branch, tree will always be allowed to choose best split. As such, it should be deterministic
        # when provided the same data, irrespectively of post-pruning parameter ccp_alpha.
        unpruned_model = DecisionTreeClassifier(max_depth = max_depth).fit(features_train, labels_train) 
        
        current_fold_results = {alpha: {} for alpha in post_pruning_alphas}
        
        for alpha in post_pruning_alphas:
            # Instead of retraining multiple times the same tree and then post-pruning each time, train a single time,
            # deepcopy the instance and prune it once per each post-pruning parameter and compute scores. 
            model = deepcopy(unpruned_model)
            model.set_params(ccp_alpha = alpha)
            model._prune_tree()
            
            for key, X, y in [('train_scores', features_train, labels_train),
                              ('val_scores'  , features_val  , labels_val)]:
            
                preds = model.predict(X)

                current_fold_results[alpha][key] = _scores_classification(y, preds)
            
        results.append(current_fold_results)
            
    return results


@utils.cache_to_file_pickle("cross_validation-get_pruning_alphas_impurities", ignore_kwargs = ['X_train'])
def get_pruning_alphas_impurities(X_train, y_train, problem_type, downsampling_factor, max_depth):
    
    y_train = np.array(y_train)
    
    if problem_type == 'regression':
        tree = DecisionTreeRegressor(max_depth = max_depth)
        stratifier = utils.get_percentile_stratifier(y_train)
    
    elif problem_type == 'classification':
        tree = DecisionTreeClassifier(max_depth = max_depth)
        stratifier = y_train

    else:
        raise ValueError("Parameter 'problem_type' must be either 'regression' or 'classification'.")
        
    
    if downsampling_factor > 1:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, 
                                                  train_size = 1 / downsampling_factor, 
                                                  shuffle = True, 
                                                  stratify = stratifier)
        
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    return path.ccp_alphas, path.impurities