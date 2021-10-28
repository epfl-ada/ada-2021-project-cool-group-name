import os
import pickle
import requests
import re

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
        
    
def cache_to_file_pickle(filename, cache_dir = 'Cache', ignore_kwargs = None):
    # Warning: filename should follow convention: "{class name}-{method name}" for class methods and "function-{function name}" for
    # functions declared outside classes.
    
    # This function is not thread-safe!
    
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


def get_labels_of_wikidata_ids(ids):    
    sparql_query = """
        SELECT ?item ?itemLabel
        WHERE {
            VALUES ?item {""" + ' '.join(['wd:' + elem for elem in ids]) + """}
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
    """

    url = 'https://query.wikidata.org/sparql'

    data = requests.get(url, params = {'format': 'json', 'query': sparql_query}).json()
    
    items_and_labels = data['results']['bindings']
    
    mapping = {}
    for result in items_and_labels:
        item = re.sub(r".*[#/\\]", "", result['item']['value'])
        label = result['itemLabel']['value']
        mapping[item] = label
    
    return mapping
