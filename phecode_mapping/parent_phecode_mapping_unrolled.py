import json
import pandas as pd

phecode_definitions = pd.read_csv('phecode_definitions1.2.csv')
# here we use all phecodes in integer level as parent phecode, 586
# 305 phecode does not exist, it only has 305.2 and 305.21. Excluding 305, we have 585 phecodes.
parent_phecodes = list(phecode_definitions.phecode.astype(int).unique()) # 586
phecode_icd9_map = pd.read_csv('phecode_icd9_map_unrolled.csv')
phecode_icd9_map.phecode = phecode_icd9_map.phecode.astype(int) # from float to string
phecode_icd9_map.drop_duplicates(inplace=True)
unique_icd9 = list(phecode_icd9_map.icd9.unique()) # 13707
unique_phecode = list(phecode_icd9_map.phecode.unique())# 544, less than 586 , some phecodes may not appear? check
# now each icd only maps to a single phecode, thus we obtain 13707 icd codes for 544 phecodes
phecode_icd_items = list(zip(phecode_icd9_map.phecode, phecode_icd9_map.icd9)) # get all (phecode, icd9) pairs
phecode_icd_dict = {} # each phecode is a key, the value corresponds to a key is [ICD9, ... , ICD9]
for pc, icd in phecode_icd_items:
    if pc not in phecode_icd_dict.keys():
        phecode_icd_dict[pc] = [icd]
    else:
        phecode_icd_dict[pc].append(icd)
with open('parent_phecode_icd_dict_unrolled.json', 'w') as fp:
    json.dump(phecode_icd_dict, fp)
