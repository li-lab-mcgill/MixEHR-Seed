# compute the initialized expected sufficient statistics for each modalities based on topic prior alpha
# todo: read from the metedata instead of hard writing
import torch
from corpus import Corpus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds_topic_matrix = torch.load("../phecode_mapping/seed_topic_matrix.pt", map_location=device) # get seed word-topic mapping, V x K matrix
topic_prior_alpha = torch.load("./file/topic_prior_alpha.pt", map_location=device)  # get topic prior alpha, D X K matrix
c = Corpus.read_corpus_from_directory('../store/', 'corpus.pkl') # read corpus file
print(c.V)

exp_n_icd = torch.zeros(c.V[0], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_s_icd = torch.zeros(c.V[0], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device) # default guided modality is 0
for d_i, doc in enumerate(c.dataset):
    print(d_i)
    doc_id = doc.doc_id
    for word_id, freq in doc.words_dict[0].items(): # word_index v and freq
        # update seed words
        exp_s_icd[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1 # * 0.7
        exp_n_icd[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1 # * 0.3
        # update regular words
        exp_n_icd[word_id] += (1-seeds_topic_matrix)[word_id] * freq * topic_prior_alpha[d_i]

exp_n_med = torch.zeros(c.V[1], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_n_cpt = torch.zeros(c.V[2], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_n_drg = torch.zeros(c.V[3], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_n_lab = torch.zeros(c.V[4], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_n_note = torch.zeros(c.V[5], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
for d_i, doc in enumerate(c.dataset):
    print(d_i)
    doc_id = doc.doc_id
    for word_id, freq in doc.words_dict[1].items():
        exp_n_med[word_id] += topic_prior_alpha[d_i] * freq
    for word_id, freq in doc.words_dict[2].items():
        exp_n_cpt[word_id] += topic_prior_alpha[d_i] * freq
    for word_id, freq in doc.words_dict[3].items():
        exp_n_drg[word_id] += topic_prior_alpha[d_i] * freq
    for word_id, freq in doc.words_dict[4].items():
        exp_n_lab[word_id] += topic_prior_alpha[d_i] * freq
    for word_id, freq in doc.words_dict[5].items():
        exp_n_note[word_id] += topic_prior_alpha[d_i] * freq

torch.save(exp_n_icd, "./init_tokens/init_exp_n_icd.pt")
torch.save(exp_s_icd, "./init_tokens/init_exp_s_icd.pt")
torch.save(topic_prior_alpha, "./init_tokens/init_exp_m.pt")
torch.save(exp_n_med, "./init_tokens/init_exp_n_med.pt")
torch.save(exp_n_cpt, "./init_tokens/init_exp_n_cpt.pt")
torch.save(exp_n_drg, "./init_tokens/init_exp_n_drg.pt")
torch.save(exp_n_lab, "./init_tokens/init_exp_n_lab.pt")
torch.save(exp_n_note, "./init_tokens/init_exp_n_note.pt")
