# define data structure for corpus
import numpy as np
import pandas as pd
import pickle
import os
import logging
import sys
import time
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from utils import tokenize_phecode_icd_corpus

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Select one command', dest='cmd')

# parser process
parser_process = subparsers.add_parser('process', help="Transform MixEHR raw data")
parser_process.add_argument("-n", "--max", help="Maximum number of observations to select", type=int, default=None)

# parser split
parser_split = subparsers.add_parser('split', help="Split data into train/test")
parser_split.add_argument("-tr", "--testing_rate", help="Testing rate. Default: 0.3", type=float, default=0.3)

# default arguments
parser.add_argument('input', help='Directory containing input data')
parser.add_argument('output', help='Directory where processed data will be stored')


class Corpus(Dataset):
    def __init__(self, docs, modaltiy_list, V, C):
        logger.info("Creating corpus...")
        self.dataset = docs # a list of Document objects
        self.modalities = modaltiy_list # a list of modalities
        self.D = len(docs)
        self.V = V # V is the vocab of all words (regular and seed) for different modality
        self.C = C # C is the number of words in the corpus
        # maybe we should set self.M, if the number of modality is more and it is not convenient to set one by one

    def __len__(self):
        return self.D

    def __getitem__(self, index):
        '''
        Generate one sample for dataLoader
        '''
        doc_sample = self.dataset[index]
        return doc_sample, index

    @staticmethod
    def __collate_model__(batch):
        '''
        Returns a batch for each iteration of the DataLoader
        '''
        docs, indexes = zip(*batch) # list of documents and indices
        # list of docs in minibatch, indexes of docs in minibatch, docs' total number of words in a minibatch
        return list(docs), np.array(indexes), [doc.Cd for doc in docs]

    @staticmethod
    def build_from_GDTM_fileformat(modaltiy_list, path_dict, column_dict, store_path=None):
        '''
        Reads a multi_modal EHR data and return a Corpus object.
        :param modaltiy_list: the list of modality we want to train the model
        :param path_dict: the dict of dataframe path for each modality
        :param column_dict: the dict of which column is defined as word for each modality
        :param store_path: store output Corpus object.
        '''
        def __read_docs__(data_list, vocab_list, modaltiy_list, column_dict, guided_modaltiy=0):
            training = {}
            for m, modality_name in enumerate(modaltiy_list):
                print(m, modality_name)
                data_m = data_list[m]
                vocab_m = vocab_list[m]
                column = column_dict[modality_name]
                num_records = data_m.shape[0]
                with tqdm(total=num_records) as pbar:
                    for i, row in enumerate(data_m.iterrows()): # for each record, append to its document
                        row = row[1]
                        doc_id = row['SUBJECT_ID']
                        word_id = vocab_m[row[column]]
                        freq = row['FREQ']
                        if doc_id not in training:
                            training[doc_id] = Corpus.Document(doc_id, modality_num=len(modaltiy_list))
                        doc = training[doc_id]
                        doc.append_record(word_id, freq, m)
                        pbar.set_description("%.4f  - document(%s), word(%s)" % (100 * (i + 1) / num_records, doc_id, word_id))
                        pbar.update(1)
            return training

        def __store_data__(toStore, corpus):
            if not os.path.exists(toStore):
                os.makedirs(toStore)
            corpus_file = os.path.join(toStore, "corpus.pkl")
            logger.info("Saving: \n\t%s" % (corpus_file))
            pickle.dump(corpus, open(corpus_file, "wb"))
            logger.info("Data stored in %s" % toStore)

        # For each modality, read data and count records
        data_list = [] # the dataframe for each modality
        C_list = [] # the number of records in the corpus for each modality
        for m, modality_name in enumerate(modaltiy_list):
            data = pd.read_csv(path_dict[modality_name])
            print(data)
            data_list.append(data)
            C_list.append(data.FREQ.to_numpy().sum())

        # For each modality, map words into index space
        vocab_list = []
        for modality_name, column in column_dict.items():
            print('export ' + modality_name + ' modality')
            m = modaltiy_list.index(modality_name)
            if modality_name == 'icd':
                # for the guided modality of ICD
                # phecode_ids: key is phecode, value is the mapped index of phecode from 1 to K-1, K is 1502
                # icd_vocab_ids: key is icd, value is the mapped index of icd from 1 to V-1, V is 6807
                # tokenized_phecode_icd is dict {mapped phecode: [mapped ICD codes]}, len(key) is 1502, len(values) is 6807, other are regular words
                data = data_list[m]
                phecode_ids, icd_vocab_ids, tokenized_phecode_icd = tokenize_phecode_icd_corpus(data)
                vocab_list.append(icd_vocab_ids)
                with open('mapping/icd_vocab_ids.pkl', 'wb') as handle:
                    pickle.dump(icd_vocab_ids, handle)
                with open('mapping/phecode_ids.pkl', 'wb') as handle:
                    pickle.dump(phecode_ids, handle)
                with open('mapping/tokenized_phecode_icd.pkl', 'wb') as handle:
                    pickle.dump(tokenized_phecode_icd, handle)
            else:
                # for the unguided modality of other modalities
                vocab_ids = {}  # key is word, value is the mapped index of word from 1 to V-1
                data = data_list[m]
                word_list = data[column].unique()
                for i, word in enumerate(word_list):
                    vocab_ids[word] = i
                vocab_list.append(vocab_ids)
                with open('mapping/' + modality_name +'_vocab_ids.pkl', 'wb') as handle:
                    pickle.dump(vocab_ids, handle)
        print("finish exporting mapping")

        # Process and read documents
        # modality is 6 for MIMIC data and the first modality is guided by default
        print('read multi-modal EHR data')
        dataset = __read_docs__(data_list, vocab_list, modaltiy_list, column_dict, guided_modaltiy=0)
        corpus = Corpus([*dataset.values()], modaltiy_list, [len(vocab) for vocab in vocab_list], C_list) # Set data to Corpus object
        logger.info(f'''
        ========= DataSet Information =========
        Documents: {len(corpus.dataset)}
        Word Tokens for each modality: {corpus.V}
        ======================================= 
        ''')
        if store_path:
            __store_data__(store_path, corpus)
        return corpus

    @staticmethod
    def split_train_test(corpus, split_rate, toStore):
        '''
        train-test split for documents in the corpus object
        '''
        assert split_rate >= .0 and split_rate <= 1., "specify the rate for splitting training and test. e.g 0.8 = 80% for testing"

        def __store_data__(toStore, corpus):
            if not os.path.exists(toStore):
                os.makedirs(toStore)
            corpus_file = os.path.join(toStore, "corpus.pkl")
            logger.info("Saving: \n\t%s" % (corpus_file))
            pickle.dump(corpus, open(corpus_file, "wb"))
            logger.info("Data stored in %s" % toStore)

        def __split__(train_size, corpus):
            documents = [] # initialize to store train documents
            corpus_list = [None, None]
            splitted = False
            M = len(corpus.modalities) # modality number
            C = [0 for m in range(M)]
            index = 0 # set index to zero for train set, doc_id from 0 to D-1 ，original code is -1, need to check
            dbar = tqdm(corpus)
            for doc, _, in dbar:
                dbar.set_description("Processing document %s" % doc.doc_id) # check description
                doc.doc_id = index # doc_id from 0 to D-1
                index += 1
                for m, Cd_m in enumerate(doc.Cd):
                    C[m] += Cd_m
                documents.append(doc)
                if index == train_size and not splitted:
                    corpus_list[0] = Corpus(documents, corpus.modalities, corpus.V, C) # obtain train set
                    documents = [] # initialize to store test documents
                    index = 0 # set index to zero for test set
                    C = [0 for m in range(M)]
                    splitted = True
            corpus_list[1] = Corpus(documents, corpus.modalities, corpus.V, C) # obtain test set
            return tuple(corpus_list)

        train_size = corpus.D - int(split_rate * corpus.D)
        train, test = __split__(train_size, corpus)
        # store data
        __store_data__(os.path.join(toStore, 'train'), train)
        __store_data__(os.path.join(toStore, 'test'), test)
        logger.info("Training size: %s\nTesting size: %s\n" % (train_size, corpus.D - train_size))

    @staticmethod
    def read_corpus_from_directory(path, corpus_name='corpus.pkl'):
        '''
        Reads existed data
        '''
        corpus_file = os.path.join(path, corpus_name)
        corpus = pickle.load(open(corpus_file, "rb"))
        return corpus

    @staticmethod
    def generator_mini_batch(corpus, batch_size):
        generator = DataLoader(corpus, batch_size=batch_size, shuffle=True, collate_fn=Corpus.__collate_model__)
        return generator

    @staticmethod
    def generator_full_batch(corpus):
        generator = DataLoader(corpus, batch_size=len(corpus), shuffle=True, collate_fn=Corpus.__collate_model__)
        return generator

    class Document(object):
        def __init__(self, doc_id, words_dict:dict=None, modality_num=1):
            '''
            Create a new document.
            '''
            self.doc_id = doc_id # index of document, doc_id for train set and test set starts from 0
            self.words_dict = [{} for m in range(modality_num)] # key: word_id, value: frequency
            self.Cd = [0 for m in range(modality_num)]  # number of words of a document

        def append_record(self, word_id, freq, modality):
            '''
            Append a record to a document's words dict
            '''
            self.words_dict[modality][word_id] = freq # key is index of word in vocabulary, value if its frequency
            self.Cd[modality] += freq # add freq to Cd

        def __repr__(self):
            return "<Document object (%s)>" % self.__str__()

        def __str__(self): # print Document object will return this string
            # return "Document id: (%s), Words %s, Count %s" % (self.doc_id, sum([len(words_dict) for words_dict in self.words_dict]), sum([Cd for Cd in self.Cd]))
            return "Document id: (%s), Words %s, Count %s" % (self.doc_id, [len(words_dict) for words_dict in self.words_dict], [Cd for Cd in self.Cd])

def run(args):
    cmd = args.cmd
    BASE_FOLDER = args.input
    STORE_FOLDER = args.output
    print(BASE_FOLDER)
    print(STORE_FOLDER)
    if cmd == 'process':
        metadata = pd.read_csv(os.path.join(BASE_FOLDER, 'metadata.csv'), index_col='index')
        print(metadata)
        modality_list = metadata.index.tolist()
        path_dict = metadata['path'].to_dict() # the data path for each modality
        column_dict = metadata['word_column'].to_dict() # the column of csv file is defined as word for each modality
        print(modality_list)
        print(path_dict)
        print(column_dict)
        Corpus.build_from_GDTM_fileformat(modality_list, path_dict, column_dict, STORE_FOLDER)

    elif cmd == 'split':
        testing_rate = args.testing_rate
        c = Corpus.read_corpus_from_directory(BASE_FOLDER, 'corpus.pkl')
        print(c.V)
        Corpus.split_train_test(c, testing_rate, STORE_FOLDER)

if __name__ == '__main__':
    run(parser.parse_args(['process', '-n', '150', './data/', './store/']))
    # run(parser.parse_args(['split', 'store/', 'store/']))

