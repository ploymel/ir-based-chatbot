from elasticsearch import Elasticsearch, helpers
import json
import pandas as pd
import csv
from time import time

qa_mapping = {
    "properties": {
        
        "question": {
            "type": "text",
            "analyzer": "english"
        },
        "answer": {
            "type": "text"
        }
    }
}

renames_key = {
    'question': 'question',
    'answer': 'answer'
}


def load2_elasticsearch():
    index_name = 'qnap_qa'
    type = 'one_to_one'
    es = Elasticsearch()

    # Create Index
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
    else:
        es.indices.delete(index=index_name)
        print('Index deleted!')
        es.indices.create(index=index_name)
    print('Index created!')

    # Put mapping into index
    if not es.indices.exists_type(index=index_name, doc_type=type):
        es.indices.put_mapping(
            index=index_name, doc_type=type, body=qa_mapping, include_type_name=True)
    print('Mappings created!')

    # Import data to elasticsearch
    with open('data/All.csv', 'r',encoding='utf-8-sig') as outfile:
        reader = csv.DictReader(outfile)
        success, _ = helpers.bulk(es, reader, index=index_name, doc_type=type, ignore=400)
        print('success: ', success)

load2_elasticsearch()