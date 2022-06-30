import json
import pandas as pd

# here we use all phecodes in decimal level as parent phecode, 1360
phecode_icd9_map = pd.read_csv('phecode_icd9_rolled.csv')
# unique_all_phecode = phecode_icd9_map.loc[(phecode_icd9_map['Leaf'] ==1)].PheCode.unique() # 1360
# unique_icd9 = list(phecode_icd9_map.ICD9.unique()) # 15558
# now each icd only maps to a single phecode, thus we obtain 15558 icd codes for 586 phecodes
phecode_icd_items = list(zip(phecode_icd9_map.PheCode, phecode_icd9_map.ICD9)) # get all (phecode, icd9) pairs
phecode_icd_dict = {} # each phecode is a key, the value corresponds to a key is [ICD9, ... , ICD9]
for pc, icd in phecode_icd_items:
    if pc not in phecode_icd_dict.keys():
        phecode_icd_dict[pc] = [icd]
    else:
        phecode_icd_dict[pc].append(icd)
with open('all_phecode_icd_dict_rolled.json', 'w') as fp:
    json.dump(phecode_icd_dict, fp)
