# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:43:35 2021

"""

import requests
import re

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
