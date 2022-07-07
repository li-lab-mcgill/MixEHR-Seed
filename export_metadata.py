import pandas as pd
import os

BASE_FOLDER = './data/'
path_icd = os.path.join(BASE_FOLDER, 'icd_toy_data.csv') # remove the ICD codes that do not correspond to PheCodes
path_pres = os.path.join(BASE_FOLDER, 'pres_toy_data.csv')
path_cpt = os.path.join(BASE_FOLDER, 'cpt_toy_data.csv')
path_drg = os.path.join(BASE_FOLDER, 'drg_toy_data.csv')
path_lab = os.path.join(BASE_FOLDER, 'lab_toy_data.csv')
path_note = os.path.join(BASE_FOLDER, 'note_toy_data.csv')
path_dict = {'icd': path_icd, 'pres': path_pres, 'cpt': path_cpt, 'drg': path_drg, 'lab': path_lab, 'note': path_note}
column_dict = {'icd': 'ICD9', 'pres': 'COMPOUND_ID', 'cpt': 'ICD9_CODE', 'drg': 'COMPOUND_ID', 'lab': 'LABEL',
               'note': 'TERM'}  # which column is defined as word for each modality
df = pd.DataFrame([path_dict, column_dict])
df['index'] = ['path', 'word_column']
df.set_index('index', inplace=True)
print(df.transpose())
df.transpose().to_csv(os.path.join(BASE_FOLDER, 'metadata.csv'), index_label='index')
