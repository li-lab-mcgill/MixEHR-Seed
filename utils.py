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
    and export the seed ICD - PheCodes mapping as torch tensor object

    Note: we have two option, one option is only use the seed ICD, another is including regular words (do not map to PheCodes)
    For MIMIC data, the full ICD vocab is 6696 where 5855 can be found associated with PheCodes
    If we use the first option, V = S = 5855
    If we use the second option, V = 6696, S = 5855

    :return: mapped_phecode, mapped_icd, tokenized_phecode_icd
    '''
    corpus_data_dropna = corpus_data.dropna() # only keep the ICD that finds map with PheCodes
    ICD_list = corpus_data.ICD9.unique()
    seed_ICD_list = corpus_data_dropna.ICD9.unique()
    phecode_list = corpus_data_dropna.PheCode.unique()
    V, S, K = len(ICD_list), len(seed_ICD_list), len(phecode_list) # V 12156, S 6954, K 1548
    print(V, S, K)

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
    mapped_phecode = {} # key is phecode, value is the mapped index of phecode from 1 to K-1, K is 1620 for MIMIC full data
    for i, code in enumerate(phecode_list):
        mapped_phecode[code] = i
    mapped_icd = {} # key is icd, value is the mapped index of icd from 1 to V-1, V is 6696 but S is 5855 for MIMIC full data
    for i, icd in enumerate(ICD_list):
        mapped_icd[icd] = i
    tokenized_phecode_icd = {mapped_phecode[key]: [mapped_icd[ICD] for ICD in value] for key, value in
                             phecode_icd_dict.items()}

    # generate and save phecode ICD mapping as a torch matrix
    seeds_topic_matrix = torch.zeros(V, K, dtype=torch.int)
    for k, w_l in tokenized_phecode_icd.items():
        for w in w_l:
            seeds_topic_matrix[w, k] = 1
    # print(seeds_topic_matrix.shape) # V, K, V is 6696 if do not drop, and V = S is 5855 if drop nan
    # print(seeds_topic_matrix.sum()) # 7262 as 5855 ICD are seed words across all topics, one ICD can only be mapped to one phecodes
    torch.save(seeds_topic_matrix, "./phecode_mapping/seed_topic_matrix.pt")
    return mapped_phecode, mapped_icd, tokenized_phecode_icd

