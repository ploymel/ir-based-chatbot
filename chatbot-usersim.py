from elasticsearch import Elasticsearch
import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import requests
from build_transition_matrix import get_transition_prob
import os

def IRsystemResult(query, q_id=None):
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

    for i in range(len(text)):
        if q_id != None:
            if int(q_id) == int(text[i]['_source']['f_id']):
                continue
        questions.append(text[i]['_source']['question'])
        answers.append(text[i]['_source']['answer'])
        c_ids.append(text[i]['_source']['f_id'])
        
    return questions, answers, c_ids

def sentenceBert(query, ir_ques, ir_ans, c_ids, q_id):
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
    with open('log.txt', 'w', encoding="utf-8") as file:
        file.writelines(history)
    
    # YOUR DA MODEL HERE!!
    resp = requests.post("http://140.115.54.35:5000/predict", files={"file": open('log.txt','rb')})
    question_tag = resp.json()['tags']
    # Sort transition probability using DA tags

    same_id = False
    for item in all_sentence_combinations:
        if c_ids[item[1]] == picked_conversation_id:
            same_id = True
            qid = item[1]
            picked_conversation_id = c_ids[qid]
            break
    if same_id == False:
        # confident == 1
        temp = []
        for item in all_sentence_combinations:
            if round(item[0].item(), 5) == 1 and int(c_ids[item[1]]) != q_id:
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
                with open('log.txt', 'w', encoding="utf-8") as file:
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

def read_data():
    files = os.listdir('QNAP')
    raw_data_msg = []
    raw_data_tag = []
    for fname in files:
        df = pd.read_csv('QNAP/' + fname, header=None)
        for _, row in df.iterrows():
            raw_data_msg.append(row[1])
            raw_data_tag.append(row[2])

    # read pairs data
    df_pair = pd.read_csv('all-qnap-reply-to-labeled.csv')
    df_pair = df_pair.drop_duplicates(['f_id', 'question'])
    pairs = {}
    for _, row in df_pair.iterrows():
        if row['question'] in raw_data_msg and row['answer'] in raw_data_msg:
            qid = raw_data_msg.index(row['question'])
            # aid = raw_data_msg.index(row['answer'])
            # pairs.append([raw_data_tag[qid], raw_data_tag[aid]])
            if row['f_id'] in pairs:
                pairs[row['f_id']].append([row['question'], raw_data_tag[qid]])
            else:
                pairs[row['f_id']] = [[row['question'], raw_data_tag[qid]]]

    return pairs

def generate_new_query(query, ir_ques, c_ids, c_id, raw_msg, raw_tag):
    passage_embedding = model.encode(ir_ques)
    query_embedding = model.encode(query)
    
    cos_sim = util.pytorch_cos_sim(query_embedding, passage_embedding)

    all_sentence_combinations = []

    for i in range(len(cos_sim[0])-1):
        all_sentence_combinations.append([cos_sim[0][i],i])

    #Sort list by the highest cosine similarity score
    all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

    best_item = None
    best_score = 0

    for item in all_sentence_combinations:
        if int(c_ids[item[1]]) == int(c_id):
            continue
        with open('log.txt', 'w') as file:
            file.writelines(ir_ques[item[1]] + '\n')
        resp = requests.post("http://140.115.54.35:5000/predict", files={"file": open('log.txt','rb')})
        q_tag = resp.json()['tags']
        qid = raw_msg.index(query)
        g_tag = raw_tag[qid]
        # if q_tag == g_tag:
        #     return ir_ques[item[1]], c_ids[item[1]]
        if item[0].item() > best_score:
            best_score = item[0].item()
            best_item = item
    
    return ir_ques[best_item[1]], c_ids[best_item[1]]

def get_msg_da(pairs):
    files = os.listdir('QNAP')
    raw_msg = []
    raw_tag = []
    for fname in files:
        df = pd.read_csv('QNAP/' + fname, header=None)
        for _, row in df.iterrows():
            if row[0] == 'client':
                if type(row[1]) == str:
                    msg = row[1]
                    for p in pairs:
                        for item in pairs[p]:
                            if item[0] == msg and msg != '':
                                raw_msg.append(msg)
                                raw_tag.append(row[2])
                                break

    return raw_msg, raw_tag


if __name__ == '__main__':
    model = SentenceTransformer('msmarco-distilbert-base-v3')
    history = []
    picked_conversation_id = None
    transition_matrix = get_transition_prob()
    threshold = 0.6
    
    # read conversation
    data = read_data()
    raw_msg, raw_tag = get_msg_da(data)
    c_id = int(input("conversation ID: "))
    for item in data[c_id]:
        # generate new query from original query
        # get query candidate
        ir_ques, _, ir_c_ids = IRsystemResult(item[0])
        gen_query, q_c_id = generate_new_query(item[0], ir_ques, ir_c_ids, c_id, raw_msg, raw_tag)
        print('client: {}, c_id: {}'.format(gen_query, q_c_id))
        # generate new query from original query
        ir_ques, ir_ans, ir_c_ids = IRsystemResult(gen_query, q_c_id)
        sentenceBert(gen_query, ir_ques, ir_ans, ir_c_ids, q_c_id)