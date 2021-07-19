import os
import numpy as np
import pandas as pd

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
    pairs = []
    for _, row in df_pair.iterrows():
        if row['f_id'] in [0, 5, 11, 27, 32]:
            continue
        if row['question'] in raw_data_msg and row['answer'] in raw_data_msg:
            qid = raw_data_msg.index(row['question'])
            aid = raw_data_msg.index(row['answer'])
            pairs.append([raw_data_tag[qid], raw_data_tag[aid]])

    return pairs

def count_transition(data):
    transition_counter = {
        'Feedback': {},
        'Statement': {},
        'Commissive': {},
        'Directive': {},
        'SetQ': {},
        'PropQ': {},
        'ChoiceQ': {},
        'Salutation': {},
        'Apology': {},
        'Thanking': {}
    }
    # loop through data
    for pair in data:
        if pair[1] in transition_counter[pair[0]]:
            transition_counter[pair[0]][pair[1]] += 1
        else:
            transition_counter[pair[0]][pair[1]] = 1

    return transition_counter

def gen_transition_prob(transition):
    for key in transition:
        # count total utters
        total = 0
        for t_key in transition[key]:
            total += transition[key][t_key]
        # calculate prob
        for t_key in transition[key]:
            transition[key][t_key] = transition[key][t_key]/total

    sorted_transition = {}
    for key in transition:
        sorted_keys = []
        for t_key in transition[key]:
            sorted_keys.append([t_key, transition[key][t_key]])

        sorted_keys.sort(key=lambda x: x[1], reverse=True)
        sorted_transition[key] = sorted_keys
    
    return sorted_transition

def get_transition_prob():
    pairs = read_data()
    transition = count_transition(pairs)
    transition = gen_transition_prob(transition)

    return transition

if __name__ == '__main__':
    transition = get_transition_prob()
    print(transition['Feedback'])