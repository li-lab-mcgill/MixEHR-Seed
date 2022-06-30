import json
import torch
import numpy as np


def logsumexp(x):
    x = torch.tensor(x)
    c = torch.max(x)
    return c + torch.log(torch.sum(torch.exp(x - c)))

def keystoint(x):
    '''
    convert key to int datatype
    '''
    return {float(k): v for k, v in x.items()}

def tokenize_phecode_icd_corpus(corpus_data):
    '''
    tokenization, map phecode and icd code from 0 to K-1/V-1 for a given corpus
    we include regular words for all seed topics in this function
    :return: mapped_phecode, mapped_icd, tokenized_phecode_icd
    '''
    ICD10_list = corpus_data.ICD9.unique()
    phecode_list = corpus_data.PheCode.unique()
    V, K = len(ICD10_list),  len(phecode_list) # V 12156, S 6954, K 1548
    print(ICD10_list)
    print(phecode_list)
    print(V, K)

    phecode_icd_df = corpus_data.groupby(['ICD9', 'PheCode']).size().reset_index().rename(columns={0: 'count'})
    phecode_icd_dict = {}
    for index, row in phecode_icd_df.iterrows():
        if row['PheCode'] not in phecode_icd_dict.keys():
            phecode_icd_dict[row['PheCode']] = [row['ICD9']]
        else:
            phecode_icd_dict[row['PheCode']].append(row['ICD9'])
    # print((phecode_icd_dict))
    # print(len(phecode_icd_dict.items()))
    # S_test, K_test = [], []
    # for k, v in phecode_icd_dict.items():
    #     K_test.append(k)
    #     S_test.extend(v)
    # print(len(set(S_test)), len(K_test))

    # tokenization for ICD codes and phecodes which appear in corpus
    mapped_phecode = {} # key is phecode, value is the mapped index of phecode from 1 to K-1, K is 1548
    for i, code in enumerate(phecode_list):
        mapped_phecode[code] = i
    mapped_icd = {} # key is icd, value is the mapped index of icd from 1 to V-1, V is 12156 but S is 6954
    for i, icd in enumerate(ICD10_list):
        mapped_icd[icd] = i
    tokenized_phecode_icd = {mapped_phecode[key]: [mapped_icd[ICD] for ICD in value] for key, value in
                             phecode_icd_dict.items()}

    # save phecode ICD mapping in corpus as a torch matrix
    seeds_topic_matrix = torch.zeros(V, K, dtype=torch.int) # 12156 X 1548
    for k, w_l in tokenized_phecode_icd.items():
        for w in w_l:
            seeds_topic_matrix[w, k] = 1
    # print(seeds_topic_matrix.sum()) # 7262 as 7262 words are seed words across topics, one ICD10 can map to multiple phecodes
    torch.save(seeds_topic_matrix, "./phecode_mapping/seed_topic_matrix.pt")
    return mapped_phecode, mapped_icd, tokenized_phecode_icd

