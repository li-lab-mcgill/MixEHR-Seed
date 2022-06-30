# MixEHR-Seed

MixEHR-Seed is a seed-guided Bayesian topic model that fits large-scale, longitundinal, multi-modal EHR data with thousands of phenotypic topics. 

In the seed-guidance, each phenotypic topic is represented as two distributions: (1)a seed-topic distribution over only its set of seed words;
(2) a regular-topic distribution over the entire vocabulary.

Moreover, by associating the topic mixture of each patient (with a certain age) with an age-dependent topic hyperparameters, we can model temporal topic progression in the population. We compute initial topic probabilities using a 2-componet Gaussian mixture model (GMM) on each PheCode count and then used the GMM-inferred posteriors as the initialized topic hyperparameters to guide topic inference.

We generalize our model to multi-modality by the incorporation of diverse types of EHR data. We only guide topic inference in the ICD modality using the expert knowledge from the PheWAS catalog (https://phewascatalog.org/phecodes). It defines each PheCode as a set of ICD codes, which we treat as the seeds for the corresponding phenotypic topic.

To learn our model, we devise a hybrid Bayesian inference methodology in a stochastic manner. We infer the seed-guidance topic assignments by collapsed variational mean-field inference, while infer the age-dependent topic hyperparameters by an amortized inference using a LSTM network. We can  compute the collapsed variables
topic mixture memberships \theta, regular topic distributions \phi^r, seed topic distributions \phi^s with the respective variational expectations.

The proababilistic graphical model of MixEHR-S is shown:

<<<<<<< HEAD
<img src="https://github.com/li-lab-mcgill/MixEHR-Seed/blob/main/figures/PGM.jpg" width="500" height="350">
=======
<img src="https://github.com/li-lab-mcgill/MixEHR-Seed/blob/main/figures/PGM.jpg" width="920" height="350">
>>>>>>> 625194aabdd91d62e141da666691a604da1c5f7e


# Relevant Publications

This published code is referenced from: 
<<<<<<< HEAD
Ziyang Song, Yuanyi Hu, Aman Verma, David Buckeridge, and Yue Li. 2022.
Automatic Phenotyping by a Seed-guided Topic Model. In Proceedings of the
28th ACM SIGKDD Conference On Knowledge Discovery and Data Mining,
August 14–18, 2022, Washington, DC. ACM, New York, NY, USA, 12 pages.
https://doi.org/10.1145/1122445.1122456
=======
Ziyang Song, Yuanyi Hu Aman Verma, David Buckeridge, and Yue Li*. (2022) Automatic phenotyping by a seed-guided topic model. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’22), August 14–18, 2022, Washington, DC, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3534678.3542675 (KDD HealthDay2022 Best Paper award)
>>>>>>> 625194aabdd91d62e141da666691a604da1c5f7e


# Dataset

We evaluated MixEHR-S on the extracted clinical dataset from the PopHR, UKB, and MIMIC-III database. 

For these datasets, we organize each data type of EHRs into one single input file (such as ICD codes). Moreover, the temporal information (such as a patient's age group) is listed in a separate file. 

In the path "MixEHR_Seed/data/", we have extracted a toy data with 1000 rows from MIMIC-III database including ICD and medication modaltiy.

- ICD.csv with header: patient ID,	ICD code,	PheCode, frequency


                            Headers:SUBJECT_ID,ICD9,PheCode,FREQ


- medication.csv with header: patient ID,	medication code, frequency


                            Headers:SUBJECT_ID,COMPOUND_ID,FREQ


- time.csv : patient ID, certain age group of patient, the document id 

If we open temporal inference, we have to add a separate time.csv that divides each patient's records into multiple documents. Morewore, we have to add an extra column for the multi-modal EHRs (such as ICD.csv) for DOC_ID. 

                            Headers:SUBJECT_ID,AGE,DOC_ID
              
              
# Code Description

## STEP 1: Process Dataset and extract seeds

The input data files include the multiple data types of EHR data. We have to transform the inputs into the built-in, readable data structure "Corpus" class. Moreover, we need extract the seed ICD codes of PheCodes from the dataset.

You can use "MixEHR_Seed/corpus.py" to transform the raw inputs into a runnable Corpus data structure and generate the seeds of phenotypic topics. 

Place dataset to the specific path "MixEHR_Seed/data/" and then run the following code:

    run(parser.parse_args(['process', '-n', '150', './data/', './store/']))
    
you also need to split the dataset into train/validation/test subset. The data path and detailed split ratio could be edited:
    
    run(parser.parse_args(['split', 'store/test/', 'store/']))
	
The extracted PheCode-ICD mapping is located at path "MixEHR_Seed/phecode_mapping/all_seed_topic_matrix.pt", where each row and column represents a word and a topic, respectively.


## STEP 2: Compute initial topic prior using GMM

For each phenotypic topic, count the frequency of its seed ICD codes under the PheCode by runing the file "MixEHR_Seed/guide_prior/get_doc_phecode.py"

For each phenotypic topic, train a 2-component GMM for the PheCode counts over all patients to have predictive probabilities (initial topic prior) by running the file "MixEHR_Seed/guide_prior/get_prior_GMM.py"

Compute the initial sufficient statistics by running the file: "MixEHR_Seed/guide_prior/get_token_counts.py"

## STEP 3: Topic Modelling

We can run "MixEHR_Seed/main.py" to perform seed-guided topic modelling on the extracted train data. 
The execution code is:

    run(parser.parse_args(['./test_store/', './result/']))
    
The topic hyperparameters of regular topics and seed topics need to fine-tune by minimizing the held-out negative log-likelihood on the validation set. We then apply MixEHR-Seed with the estiated hyperparameters on the train set. 
 

## STEP 4: Evalutions

The learned parameters are saved at "MixEHR_Seed/parameters/", we then evaluate the topic interpretability, phenotype prediction, and temporal disease progression analysis. 
    






