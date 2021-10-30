import os
import pickle
import requests
import re
import time
import bz2
import json
from math import ceil
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


def get_labels_of_wikidata_ids(ids, chunk_size = 500, wait_between_chunks_secs = 1):
    """
    Function querying Wikidata for human-readable English labels of the ids provided as parameters.
    To avoid the server refusing requests, the ids are split into chuncks of desired size and a
    different request is made for each chunk, waiting a certain amount of time between requests.
    
    Params:
         ids::[iterable]
            The Wikidata ids we want to obtain a human-readable English label of.
         chunk_size::[float]
            The number of Wikidata ids we want to send in each request. Larger values cause less requests
            to be sent and thus generally lower execution times, but also increase the chance of Wikidata
            not answering the request.
        wait_between_chunks_secs::[float]
            Number of seconds to wait between consecutive requests to Wikidata. A large enough
            value for this parameter ensures Wikidata will not stop answering requests from our machine.
            
    Returns:
        mapping::[dict]
            Dictionary with keys the Wikidata ids and values the English label for each id.
    
    """
    
    
    def request_labels_of_wikidata_ids_one_chunk(ids):
        """
        Function querying Wikidata for human-readable English labels of the ids provided as parameters.
        All labels are queried in a single request.
        
        Params:
            ids::[iterable]
                The Wikidata ids we want to obtain a human-readable English label of.
        Returns:
            items_and_labels::[list[dict]]
                List of dictionary as returned by Wikidata. Can be parsed to extract the label for of the requested ids.

        """
        
        sparql_query = """SELECT ?item ?itemLabel
                          WHERE {
                              VALUES ?item {""" + ' '.join(['wd:' + elem for elem in ids]) + """}
                              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                          }"""

        url = 'https://query.wikidata.org/sparql'

        data = requests.get(url, params = {'format': 'json', 'query': sparql_query}).json()

        items_and_labels = data['results']['bindings']

        return items_and_labels
    
    
    # Removing duplicates and coverting to list.
    ids = list(set(ids))
    
    mapping = {}
    for start_idx in range(0, len(ids), chunk_size):
        # Wait some time before sending requests to avoid spamming the server.
        time.sleep(wait_between_chunks_secs)
        
        # Query Wikidata for current chunk.
        items_and_labels = request_labels_of_wikidata_ids_one_chunk(ids[start_idx:start_idx + chunk_size])
        
        # Update mapping of ids to labels. 
        for result in items_and_labels:
            item = re.sub(r".*[#/\\]", "", result['item']['value'])
            label = result['itemLabel']['value']
            
            # Filter out qids for which label is unknown: in this case Wikidata returns the qid itself as label.
            if not str_is_qid(label):
                mapping[item] = label
            
    return mapping


def str_is_qid(string):
    return bool(re.match(r"^Q\d+$", string))
        
    
def ragged_nested_sequence_to_set(array):
    return set(np.hstack(array.ravel()))
    
    
    
def process_json_file_per_line(input_file_path,
                               func, 
                               output_file_object = None,
                               return_processed_lines_list = False,
                               print_progress_every = 1000000):
    
    # Sanity check of 'print_progress_every' parameter.
    if print_progress_every is not None:
        if not isinstance(print_progress_every, int) or print_progress_every <= 0:
            raise ValueError("Parameter 'print_progress_every' is expected to be a strictly positive integer, or None.")
    
    # Preparing variables to store processed lines in file and or in a list depending on 'output_file_object' and 
    # 'return_processed_lines_list' parameters.
    to_do_after_processing = []
    if output_file_object is not None:
        to_do_after_processing.append(lambda line: output_file_object.write((json.dumps(line) + '\n').encode('utf-8')))
        
    if return_processed_lines_list:
        processed_lines = []
        to_do_after_processing.append(processed_lines.append)
           
    # Parsing, processing and (if necessary) storing input file line by line.
    start_time = time.time()
    with bz2.open(input_file_path, 'rb') as input_file:
        
        print(f'Starting processing {input_file_path}')
        
        for i, line in enumerate(input_file):
            line = json.loads(line)
            
            processed_line = func(line)
            
            for action in to_do_after_processing:
                action(processed_line)
                
            if i > 0 and print_progress_every is not None and not i % print_progress_every:
                print(f"Processed {i} lines from {input_file_path} in {(time.time() - start_time) / 60:.3f} minutes")
        
        print(f"Finished processing {input_file_path} in {(time.time() - start_time) / 60:.3f} minutes")
        
    return processed_lines if return_processed_lines_list else None