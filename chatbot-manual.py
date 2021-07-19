from elasticsearch import Elasticsearch
import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import requests
from build_transition_matrix import get_transition_prob
import torch

def IRsystemResult(query):
    es = Elasticsearch()

    index_name = "qnap_qa"

    # Query DSL
    search_params = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["question"]
            }
        },
        "size": 50
    }
    #Search document
    result = es.search(index=index_name, body=search_params)
    result = result['hits']['hits']
    result = json.dumps(result, indent=2)
    text = json.loads(result)
    
    questions = []
    answers = []
    c_ids = []

    for i in range(50):
        questions.append(text[i]['_source']['question'])
        answers.append(text[i]['_source']['answer'])
        c_ids.append(text[i]['_source']['f_id'])
        
    return questions, answers, c_ids

def sentenceBert(query, ir_ques, ir_ans, c_ids):
    global picked_conversation_id

    passage_embedding = model.encode(ir_ques)
    query_embedding = model.encode(query)
    
    cos_sim = util.pytorch_cos_sim(query_embedding, passage_embedding)

    all_sentence_combinations = []

    for i in range(len(cos_sim[0])-1):
        all_sentence_combinations.append([cos_sim[0][i],i])

    #Sort list by the highest cosine similarity score
    all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

    # Determine DA tag
    history.append(query + '\n')
    with open('log.txt', 'w') as file:
        file.writelines(history)
    resp = requests.post("http://140.115.54.35:5000/predict", files={"file": open('log.txt','rb')})
    question_tag = resp.json()['tags']
    # Sort transition probability using DA tags

    same_id = False
    for item in all_sentence_combinations:
        if c_ids[item[1]] == picked_conversation_id:
            same_id = True
            qid = item[1]
            break
    
    if same_id == False:
        # confident == 1
        temp = []
        for item in all_sentence_combinations:
            if round(item[0].item(), 5) == 1:
                qid = item[1]
                temp.append(item)
        if len(temp) != 1:
            if len(temp) > 1:
                all_sentence_combinations = temp
            # DA help
            temp_item = 0
            temp_ans = None
            priority_da_idx = 99
            is_in_da = False
            for item in all_sentence_combinations:
                    # temp = []
                    # temp.extend(history)
                qid = item[1]
                    # temp.append(ir_ans[qid] + '\n')
                with open('log.txt', 'w') as file:
                    file.writelines(ir_ans[qid] + '\n')
                resp = requests.post("http://140.115.54.35:5000/predict", files={"file": open('log.txt','rb')})
                answer_tag = resp.json()['tags']

                    # check answer in transition or not
                t_keys = [t[0] for t in transition_matrix[question_tag]]
                if answer_tag not in t_keys:
                    continue
                else:
                    is_in_da = True
                    if t_keys.index(answer_tag) < priority_da_idx:
                        if item[0].item() < threshold:
                            continue
                        temp_item = item 
                        temp_ans = answer_tag
                        priority_da_idx = t_keys.index(answer_tag)
                
            if is_in_da == False or temp_ans == None:
                qid = all_sentence_combinations[0][1]
                picked_conversation_id = c_ids[qid]
            else:
                qid = temp_item[1]
                picked_conversation_id = c_ids[qid]

        else:
            picked_conversation_id = c_ids[qid]

    history.append(ir_ans[qid] + '\n')
    print("QNAP: {} \t".format(ir_ans[qid]))

if __name__ == '__main__':
    model = SentenceTransformer('msmarco-distilbert-base-v3')
    history = []
    picked_conversation_id = None
    transition_matrix = get_transition_prob()
    threshold = 0.6
    
    while True:
        querytext = input("client: ")
        ir_ques, ir_ans, ir_c_ids = IRsystemResult(querytext)
        sentenceBert(querytext, ir_ques, ir_ans, ir_c_ids)