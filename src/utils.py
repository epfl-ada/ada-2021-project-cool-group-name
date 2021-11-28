import os
import pickle
import requests
import re
import time
import bz2
import json
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

    
def cache_to_file_pickle(filename, cache_dir = 'Cache', ignore_kwargs = None):
    """
    Utilitary decorator allowing to store the output of the function for a given set of parameters in a binary
    file on disk and, upon calling of the function with the same set of parameters, return directly the result without
    performing all the computation again.
    
    Warning: this decorator is not thread-safe!
    
    Warning: this decorator requires to explicitely choose the filename of the binary file which will store the combinations
    of parameters and results of the function. If two functions are mapped to the same file (same cache_dir and filename),
    erroneous behaviour can occurr. For this reason, the following naming convention for the filename is proposed:
    filename = "{class name}-{method name}"     for class methods, 
               "{package name}-{function name}" for package methods,
               "function-{function name}" for functions outside packages.
               
    Warning: this decorator removes randomness in the result!
    
    Warning: by design, when using this decorator all function parameters must be passed as keyword-arguments.
    
    Warning: by design, when using this decorator all function parameters must be python standard types (int, float, str,
    list, tuple, set, dict)
               
    Params:
        filename::[str]
            Name of the binary file which will store the previously observed combinations of function parameter values and
            function outputs.
        cache_dir::[str]
            The name of the directory in which the file will be created. The same directory can be used to store the cache files
            of several functions. 
        ignore_kwargs::[tuple | list | None]
            The names of the parameters which can safely be ignored when storing result as they do not impact the function result.
            Use None if all parameters impact the function result.
    
    Returns:
        decorator::[function]
            The decorator configured to respect requested filename, cache_dir and ignore_kwargs.
    """
    
    def recursive_std_types_to_tuple(obj):
        """
        Utilitary function which converts a python standard type object passed as parameter into an hashable object in a deterministic
        manner (for exemple dealing with randomness in dictionary and set elements positions.

        Params:
            obj::[int | float | str | list | tuple | set | dict]
                Python standard type object which we want to convert into an hashable deterministic form.
            
        Returns:
            hashable_obj::[int | float | str | tuple]
                Hashable version of input object.
        """
        if isinstance(obj, (int, float, str)):
            return obj

        elif isinstance(obj, (list, tuple)):
            return tuple(recursive_std_types_to_tuple(elem) for elem in obj)

        elif isinstance(obj, set):
            return tuple(recursive_std_types_to_tuple(elem) for elem in sorted(obj))

        elif isinstance(obj, dict):
            return tuple((key, recursive_std_types_to_tuple(obj[key])) for key in sorted(obj))

        else:
            raise TypeError("Unsupported type: {}".format(type(obj)))    
    
    
    os.makedirs(cache_dir, exist_ok = True)
    cache_file_path = os.path.join(cache_dir, filename)
    
    # Sanity check of 'ignore_kwargs' parameter.
    if ignore_kwargs is not None:
        if not isinstance(ignore_kwargs, (list, tuple)) or not all(isinstance(key, str) for key in ignore_kwargs):
            raise TypeError("ignore_kwargs params must be either a list (or tuple) of strings, or None.")
    
    
    def decorator(function):
        """
        Decorator returned by cache_to_file_pickle. It takes a function as input and returns it wrapped into additional code.
        
        Params:
            function::[function]
                The function object to be decorated.
            
        Returns:
            wrapper::[function]
                The decorated function.
        """
        
        def wrapper(**kwargs):
            """
            Function called in place of original function when using decorator. This wrapper checks if the parameters passed to
            the original function have already been observed using the binary file stored on disk, and if so does not call the
            original function, instead directly returning the result. If the result is not known yet, the original function is
            called and the result stored for this combination of parameters. 
            
            Note that the warning in cache_to_file_pickle refer to this function.
            
            Params:
                kwargs::[dict]
                    The keyword-arguments passed as parameters to the original function.

            Returns:
                result::[object]
                    The result obtained by calling the original function with the kwargs set of parameters.
            """
            # Load cache if available.
            cache = {}
            if os.path.isfile(cache_file_path):
                with open(cache_file_path, 'rb') as file:
                    cache = pickle.load(file)
            
            execution_exception = None
            try:                
                # If some kwargs can be ignored as they don't change final result, do not include them in key.
                params = {key: val for key, val in kwargs.items() if key not in ignore_kwargs} if ignore_kwargs is not None else kwargs
                                
                # Making a hashable representation of kwargs dict.
                params = recursive_std_types_to_tuple(params)
            
                # If result available in cache, short-circuit computation and rewriting of cache to persistent storage.
                if params in cache:
                    return cache[params]
                
                # Otherwise, compute the function output.
                cache[params] = function(**kwargs)
 
            except Exception as e:
                execution_exception = e
                
            # If exception was thrown, it is passed through to outside of decorator (and cache is not modified in persistent storage).
            if execution_exception is not None:
                raise execution_exception
                
            # If execution was successfull, save cache to persistent storage.
            with open(cache_file_path, 'wb') as file:
                pickle.dump(cache, file)
                
            return cache[params]
        
        return wrapper
    
    return decorator



def _make_chunked_requests_wikidata(ids, sparql_query_format, value_label, chunk_size = 500, wait_between_chunks_secs = 0.1,
                                    max_attempts = 1000):
    """
    Utilitary function querying Wikidata repeatedly for some property of the ids provided as parameters.
    The query used must be provided as well as the label of the desired property used in the query. 
    To avoid loss of data due to the server refusing requests (either because a single request is too large
    or because the same machine is making requests at a too high rate), the ids are split into chuncks of
    desired size and a different request is made for each chunk, waiting a certain amount of time between requests,
    and retrying a maximum number of times if the same request keeps on failing multiple times.
    Note that this function also retries in case the query was refused due to bad syntax, in which case too large
    values of max_attempts would result in a long time to wait before an exception is thrown.
    
    Params:
        ids::[iterable]
            The Wikidata ids we want to obtain a property of.
        sparql_query_format::[str.format]
             String formatter which can simply be called with a list of space-separated qids to complete the
             sparql query.
        value_label::[str]
            Label of property used in sparql query (needed to retrieve value from bindings in json).
        chunk_size::[float]
            The number of Wikidata ids we want to send in each request. Larger values cause less requests
            to be sent and thus generally lower execution times, but also increase the chance of Wikidata
            not answering the request.
        wait_between_chunks_secs::[float]
            Number of seconds to wait between consecutive requests to Wikidata. A large enough
            value for this parameter ensures Wikidata will not stop answering requests from our machine.
        max_attempts::[int]
            The number of times to retry obtaining an answer for a single request if this keeps on failing.
            Between each attempt, wait_between_chunks_secs is waited.
            
    Returns:
        mapping::[dict]
            Dictionary with keys the Wikidata ids and values the value of the requested property for each id.
    
    """    
    
    # Removing duplicates and coverting to list.
    ids = list(set(ids))
    
    url = 'https://query.wikidata.org/sparql'
    
    mapping = {}
    for start_idx in range(0, len(ids), chunk_size):  
        # Wait some time before sending requests to avoid spamming the server.
        time.sleep(wait_between_chunks_secs)
        
        # Query Wikidata for current chunk.
        sparql_query = sparql_query_format(' '.join(f'wd:{qid}' for qid in ids[start_idx:start_idx + chunk_size]))
        
        data = None
        for attempts in range(max_attempts):
            try:
                data = requests.get(url, params = {'format': 'json', 'query': sparql_query}).json()
                
                # If success, break out of loop.
                break
            
            except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
                # In case of error, do nothing but retrying.
                pass
              
        if data is None:
            raise RuntimeError(f"Wikidata query failed more than {max_attempts} times.")
                
        # Update mapping of ids to labels. 
        for result in data['results']['bindings']:
            item = re.sub(r".*[#/\\]", "", result['item']['value'])
            value = result[value_label]['value']
            
            # Ignore qids for which value is unknown (useful if value is the qid label as in this case 
            # Wikidata returns the qid itself as label).
            if not str_is_qid(value):
                mapping[item] = value
                
    return mapping
                
           
def get_labels_of_wikidata_ids(ids, *args, **kwargs):
    """
    Function querying Wikidata for human-readable English labels of the ids provided as parameters.
    This function internally uses _make_chunked_requests_wikidata.
    
    Params:
        ids::[iterable]
            The Wikidata ids we want to obtain the label of.
        args::[tuple]
             Additional positional arguments which will be passed directly _make_chunked_requests_wikidata.
        kwargs::[dict]
             Additional keyword arguments which will be passed directly _make_chunked_requests_wikidata.

    Returns:
        mapping::[dict]
            Dictionary with keys the Wikidata ids and values the label of each id.    
    """
    sparql_query_format = """SELECT ?item ?itemLabel
                             WHERE {{
                                 VALUES ?item {{ {} }}
                                 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                             }}""".format
                                           
    return _make_chunked_requests_wikidata(ids, sparql_query_format, 'itemLabel', *args, **kwargs)
                                           

def get_link_counts_of_wikidata_ids(ids, *args, **kwargs):
    """
    Function querying Wikidata for the link counts associated to each of the ids provided as parameters.
    This function internally uses _make_chunked_requests_wikidata.
    
    Params:
        ids::[iterable]
            The Wikidata ids we want to obtain the link count of.
        args::[tuple]
             Additional positional arguments which will be passed directly _make_chunked_requests_wikidata.
        kwargs::[dict]
             Additional keyword arguments which will be passed directly _make_chunked_requests_wikidata.

    Returns:
        mapping::[dict]
            Dictionary with keys the Wikidata ids and values the link count of each id.    
    """
    sparql_query_format = """SELECT ?item ?linkcount
                             WHERE {{
                                 VALUES ?item {{ {} }}
                                 ?item wikibase:sitelinks ?linkcount.
                             }}""".format
   
    return _make_chunked_requests_wikidata(ids, sparql_query_format, 'linkcount', *args, **kwargs)


def str_is_qid(string):
    """
    Utilitary function which returns True if the string passed as parameter is of form 'Q{integer}' and False otherwise.
    
    Params:
        string::[str]
            The string we want to check the form of.
            
    Returns:
        match::[bool]
            Boolean describing whether string is of form 'Q{integer}' or not.
    """
    return bool(re.match(r"^Q\d+$", string))



def ragged_nested_sequence_to_set(array):
    """
    Utilitary function transforming an array containing sub-arrays of different sizes into a set containing each element of
    each sub-array.
    
    Params:
        array::[np.array]
            The array containing sub-arrays of different sizes.
            
    Returns:
        elements_set::[set]
            Set containing each element of each sub-array or input array.
    """
    elements_set = set()
    
    for case in array.ravel():
        if not isinstance(case, (list, tuple, np.ndarray)):
            case = [case]
            
        elements_set.update(case)
        
    return elements_set
    


def json_lines_generator(data_dir_or_path, print_progress_every = 1000000):
    """
    Utilitary function implementing a generator of lines read from the .json.bz2 files contained in data_dir_or_path (if this
    parameter points to a directory) or from the file data_dir_or_path (if this parameter points to a .json.bz2 file).
    For each file in the directory (or just the file data_dir_or_path points to) and for each line in the compressed json file, this 
    generator will read the line, decode it from json and yield the result. The generator returns when no more lines are to be read 
    from any json files in the directory (or from the single json file data_dir_or_path points to).
    
    Params:
        data_dir_or_path::[str]
            The path to either a directory containing .json.bz2 files or the path to a single .json.bz2 file.
        print_progress_every::[int]
            The frequency (in terms of number of lines between events) at which to print the number of lines yielded by the
            generator and the time elapsed until now.
            
    Yields:
        line::[dict]
            Dictionary returned by decoding each json line in the .json.bzw files in data_dir_or_path.
            
    Returns:
        None
    """
    # Sanity check of 'print_progress_every' parameter.
    if print_progress_every is not None:
        if not isinstance(print_progress_every, int) or print_progress_every <= 0:
            raise ValueError("Parameter 'print_progress_every' is expected to be a strictly positive integer, or None.")
    
    # Determine if data_dir_or_path is a dir, in which case list all .json.bz2 files in it, or if it is a single
    # file, just use it.
    if data_dir_or_path.endswith('.json.bz2'):
        input_files_paths = [data_dir_or_path]
    elif os.path.isdir(data_dir_or_path):
        subfiles = [filename for filename in os.listdir(data_dir_or_path) if filename.endswith('.json.bz2')]
        input_files_paths = [os.path.join(data_dir_or_path, filename) for filename in subfiles]
    else:
        raise ValueError("Parameter 'data_dir_or_path' is expected to be either the path to a .json.bz2 file or a directory")
    
    # Parsing and yielding lines.
    for input_file_path in input_files_paths:
        start_time = time.time()
        
        with bz2.open(input_file_path, 'rb') as input_file:

            if print_progress_every is not None:
                print(f'Starting processing {input_file_path}')

            for i, line in enumerate(input_file):
                line = json.loads(line)
                yield line

                if i > 0 and print_progress_every is not None and not i % print_progress_every:
                    print(f"Processed {i} lines from {input_file_path} in {(time.time() - start_time) / 60:.3f} minutes")

            if print_progress_every is not None:
                print(f"Finished processing {input_file_path} in {(time.time() - start_time) / 60:.3f} minutes")



@cache_to_file_pickle("utils-query_wikidata_for_linkcounts_and_labels", ignore_kwargs = ["print_progress_every"])
def query_wikidata_for_linkcounts_and_labels(data_dir, speaker_info_file_path, print_progress_every = 10000000):  
    """
    Utilitary function querying Wikidata for the link counts of all speakers in Quotebank dataset that have ambiguous QIDs
    (speaker has homonyms and as such we can't know which person it corresponds to just from the name), and also querying
    the label of all QIDs observed in the speakers info dataset (such as the labels of the occupations, nationalities, religions, ...).
    
    Params:
        data_dir::[str]
            The path to the directory containing .json.bz2 files making up the Quotebank dataset.
        speaker_info_file_path::[str]
            The path to the .parquet file containing some of the speaker informations we are interested in (gender, occupation,
            nationality, ethnicity, religion). These informations are stored as Wikidata QIDs.
        print_progress_every::[int]
            Parameter passed directly to json_lines_generator.
                  
    Returns:
        qid_labels::[dict]
            Dictionary mapping to each qid observed in the speakers info dataset an english human-readable label.
        lincounts::[dict]
            Dictionary mapping to each qid of speakers with homonyms the Wikidata link counts on qid's page.
    """
    all_speakers, speakers_needing_linkcounts = _get_unique_speakers_dataset(data_dir = data_dir, 
                                                                             print_progress_every = print_progress_every)
    
    # Load part of data extracted from Wikidata dump about speakers.
    speaker_data = get_filtered_speaker_info_data(data_dir, speaker_info_file_path, convert_to_dict = False,
                                                  columns = ['id', 'label', 'gender', 'occupation', 
                                                             'nationality', 'ethnic_group', 'religion'])
            
    # Store id-labels pairs in another variable and remove them from original dataframe.
    speaker_qid_labels = speaker_data[['id', 'label']]
    speaker_data.drop(columns = ['id', 'label'], inplace = True)
        
    # Put all qids of informations of all speakers into one single set.
    qids_needing_labels = ragged_nested_sequence_to_set(speaker_data.values)
    qids_needing_labels.remove(None)
        
    # Sanity check.
    assert all(str_is_qid(qid) for qid in qids_needing_labels)

    # Retrieve English labels for informations of all speakers. 
    qid_labels = get_labels_of_wikidata_ids(ids = qids_needing_labels)
    qid_labels = {k: v.title() for k, v in qid_labels.items()}

    # Add speakers' id-labels pairs to qid_labels.
    speaker_qid_labels = speaker_qid_labels[~speaker_qid_labels.isna().any(axis = 1)].set_index('id').to_dict('index')
    speaker_qid_labels = {k: v['label'].title() for k, v in speaker_qid_labels.items()}
    qid_labels.update(speaker_qid_labels)
    
    # Retrieve link counts for speakers for which we need it (used to decide which speaker is most likely being cited
    # amongst homonyms).
    linkcounts = get_link_counts_of_wikidata_ids(ids = speakers_needing_linkcounts)
    linkcounts = {k: int(v) for k, v in linkcounts.items()}

    return qid_labels, linkcounts


@cache_to_file_pickle("utils-_get_unique_speakers_dataset", ignore_kwargs = ["print_progress_every"])
def _get_unique_speakers_dataset(data_dir, print_progress_every = 10000000):    
    """
    Utilitary function parsing the Quotebank dataset and returning a set containing all the speaker qids present as well as
    a set containing all the speaker qids which have homonyms.
    
    Params:
        data_dir::[str]
            The path to the directory containing .json.bz2 files making up the Quotebank dataset.
        print_progress_every::[int]
            Parameter passed directly to json_lines_generator.
                  
    Returns:
        all_speakers::[set]
            Set containing the qids of all speakers in the Quotebank dataset.
        ambiguous_speakers::[dict]
            Set containing the qids of all speakers with homonyms in the Quotebank dataset.
    """
    all_speakers = set()
    ambiguous_speakers = set()
    
    for line in json_lines_generator(data_dir, print_progress_every = print_progress_every):
        line_qids_set = set(line['qids'])
        
        if len(line['qids']) > 1:
            ambiguous_speakers |= line_qids_set
            
        all_speakers |= line_qids_set
        
    return all_speakers, ambiguous_speakers


def get_filtered_speaker_info_data(data_dir, speaker_info_file_path, columns = None, convert_to_dict = True):
    """
    Utilitary function loading the speaker infos .parquet file and immediately filtering the rows and columns which are
    not useful (rows corresponding to people not in Quotebank and columns not in the columns parameter).
    
    Params:
        data_dir::[str]
            The path to the directory containing .json.bz2 files making up the Quotebank dataset.
        speaker_info_file_path::[str]
            The path to the .parquet file containing some of the speaker informations. These informations are stored as
            Wikidata QIDs.
        columns::[iterable]
            The columns of the speaker info dataframe that should not be filtered out. If None, all columns are kept.
        convert_to_dict::[bool]
            Whether after filtering the pandas dataframe containing the speaker info should be converted into a dictionary
            of form: {speaker_qid -> {column_name -> column_value_for_speaker_id}} (useful because a particular information
            about a particular speaker is faster to access this way than in a dataframe using masks).
            
    Returns:
        speaker_data::[pd.DataFrame | dict]
            Dataframe containing the speaker information from the .parquet file after filtering. 
            The dataframe itself is returned if convert_to_dict is False, otherwise it is converted into a dictionary of form
            {speaker_qid -> {column_name -> column_value_for_speaker_id}}.
    """
    all_speakers_qids, _ = _get_unique_speakers_dataset(data_dir = data_dir)
    speaker_data = pd.read_parquet(speaker_info_file_path, columns = columns)
    speaker_data = speaker_data[speaker_data['id'].isin(all_speakers_qids)]
    return speaker_data.set_index('id').to_dict('index') if convert_to_dict else speaker_data


def describe_weighted_stats(values, weights, percentiles = []):
    """
    Utilitary function computing descriptive statistics (mean, standard deviation, min, max, quartiles and additional
    percentiles passed as parameter) from a list of values and weights (number of observations) associated to each value.
    
    Params:
        values::[ordered iterable]
            The values taken by the random variable.
        weights::[ordered iterable]
            The number of times the corresponding value in values parameter was observed.
        percentiles::[iterable]
            List of additional percentiles that should be calculated for the given weighted statistical distribution.
                  
    Returns:
        stats_dict::[pd.Series]
            The statistics (mean, standard deviation, min, max, quartiles and additional percentiles passed as parameter)
            calculated for the given weighted statistical distribution.
    """    
    values, weights = list(values), list(weights)
    
    stats = DescrStatsW(values, weights = weights, ddof = 0)
    stats_dict = {'count': stats.sum_weights, 'mean': stats.mean, 'std': stats.std, 'min': stats.quantile(0.).item()}
    
    for percentile in sorted([0.25, 0.5, 0.75] + percentiles):
        stats_dict[f'{100*percentile}%'] = stats.quantile(percentile).item()
    
    stats_dict['max'] = stats.quantile(1.).item()
    
    return pd.Series(stats_dict)