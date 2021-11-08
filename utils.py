import os
import pickle
import requests
import re
import time
import bz2
import json
import pandas as pd
import numpy as np

    
def cache_to_file_pickle(filename, cache_dir = 'Cache', ignore_kwargs = None):
    # Warning: filename should follow convention: "{class name}-{method name}" for class methods and "function-{function name}" for
    # functions declared outside classes.
    
    # This function is not thread-safe!
    
    def recursive_std_types_to_tuple(obj):
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
    
    if ignore_kwargs is not None:
        if not isinstance(ignore_kwargs, (list, tuple)) or not all(isinstance(key, str) for key in ignore_kwargs):
            raise TypeError("ignore_kwargs params must be either a list (or tuple) of strings, or None.")
    
    def decorator(function):
        def wrapper(**kwargs):
            
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
            
                # If necessary, compute the function output.
                if params not in cache:
                    cache[params] = function(**kwargs)
            except Exception as e:
                execution_exception = e
                
            # If exception was thrown, it is passed through to outside of decorator.
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
    Function querying Wikidata for some property of provided ids provided as parameters.
    The query used must be provided as well as the label of the desired property used in the query.
    To avoid the server refusing requests, the ids are split into chuncks of desired size and a
    different request is made for each chunk, waiting a certain amount of time between requests.
    
    Params:
        ids::[iterable]
            The Wikidata ids we want to obtain a property of.
        sparql_query_format::[str.format]
             
         
         
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
    """Function querying Wikidata for human-readable English labels of the ids provided as parameters.
    To avoid the server refusing requests, the ids are split into chuncks of desired size and a
    different request is made for each chunk, waiting a certain amount of time between requests."""
    
    sparql_query_format = """SELECT ?item ?itemLabel
                             WHERE {{
                                 VALUES ?item {{ {} }}
                                 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                             }}""".format
                                           
    return _make_chunked_requests_wikidata(ids, sparql_query_format, 'itemLabel', *args, **kwargs)
                                           

def get_link_counts_of_wikidata_ids(ids, *args, **kwargs):
    sparql_query_format = """SELECT ?item ?linkcount
                             WHERE {{
                                 VALUES ?item {{ {} }}
                                 ?item wikibase:sitelinks ?linkcount.
                             }}""".format
   
    return _make_chunked_requests_wikidata(ids, sparql_query_format, 'linkcount', *args, **kwargs)


def str_is_qid(string):
    return bool(re.match(r"^Q\d+$", string))
        
    
def ragged_nested_sequence_to_set(array):
    elements_set = set()
    
    for case in array.ravel():
        if not isinstance(case, (list, tuple, np.ndarray)):
            case = [case]
            
        elements_set.update(case)
        
    return elements_set
    
    

def json_lines_generator(data_dir_or_path, print_progress_every = 1000000):  
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
    all_speakers, speakers_needing_linkcounts = _get_unique_speakers_dataset(data_dir = data_dir, 
                                                                             print_progress_every = print_progress_every)
    
    # Load part of data extracted from Wikidata dump about speakers.
    speaker_data = get_filtered_speaker_info_data(data_dir, speaker_info_file_path, columns = ['id', 'label', 'gender', 'occupation', 
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
    all_speakers = set()
    ambiguous_speakers = set()
    
    for line in json_lines_generator(data_dir, print_progress_every = print_progress_every):
        line_qids_set = set(line['qids'])
        
        if len(line['qids']) > 1:
            ambiguous_speakers |= line_qids_set
            
        all_speakers |= line_qids_set
        
    return all_speakers, ambiguous_speakers


def get_filtered_speaker_info_data(data_dir, speaker_info_file_path, columns = None):
    all_speakers_qids, _ = _get_unique_speakers_dataset(data_dir = data_dir)
    speaker_data = pd.read_parquet(speaker_info_file_path, columns = columns)
    speaker_data = speaker_data[speaker_data['id'].isin(all_speakers_qids)]
    return speaker_data.set_index('id').to_dict('index')