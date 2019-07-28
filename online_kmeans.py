# Aggregate ht data by category and assign category label to ul data using online k-means
# Author: Shibsankar Das
# Organization: Yodlee | TDE
# Usage: python3 online_kmeans.py

from joblib import Parallel, delayed
from scipy.spatial import distance
import pandas as pd
import numpy as np
import pickle
import math
import csv
import sys
import os

# Set csv field limit to maximum memory limit of system
csv.field_size_limit(sys.maxsize)
# Source and Destination directory path
ht_vector_src = './data/ht_data/results/final/aggregated_ht_data.pkl'
ul_desc_embeddings_src = './data/clean_data_unlabeled/results/desc_embeddings.npz'
unlabeled_nan_src = "./data/clean_data_unlabeled/results/NANs.csv"
aggregated_ul_pickle = "./data/clean_data_unlabeled/results/"

desc_embeddings_src = "./data/ht_data/results/final/desc_embeddings.npz"
# HT dict has key: tokenized description , value: category of description
ht_dict_src = "./data/ht_data/results/final/ht_description_clean.pkl"
nan_src = "./data/ht_data/results/final/NANs.csv"
aggregated_ht_pickle_dest = "./data/ht_data/results/final/"

# Load desc2vec embeddings (includes embeddings for ht and ul data), dimension = 128 and number of data members = (number of data members in ht + number of memebers in ul)
desc2vec_embeddings = np.load(desc_embeddings_src)
ht_dict = pickle.load(open(ht_dict_src,"rb"))

# Load NaN index file
nan_index = []
with open(nan_src,"r") as csvFile:
    csvReader = csv.reader(csvFile)
    for i in csvReader:
        nan_index.append(int(i[0]))

# Aggregate ht data by category
b_index = 0
vector_mapping_list = []
for d_index,description in enumerate(ht_dict):
    category = ht_dict[description]
    if d_index in nan_index:
        continue
    elif isinstance(category, (float)):
        continue
    else:
        category = ht_dict[description]
        vector_mapping_list.append({'vec':desc2vec_embeddings[b_index],'category': str(category.lower().strip())})
        b_index += 1
ht_vectors = pd.DataFrame(vector_mapping_list)
pickle.dump(ht_vectors,open(os.path.join(aggregated_ht_pickle_dest,"aggregated_ht_data.pkl"),'wb'))
ht_category_mean = ht_vectors.groupby(['category']).apply(lambda x: np.mean(x["vec"]))

def assign_label(min_index,max_index):
    assigned_label_list = []
    for index in range(min_index,max_index):
        if index not in nan_index:
            try:
                desc2vec = desc2vec_embeddings[index]
                min_distance = sys.float_info.max
                category_label = "unassigned"
                for category in ht_category_mean.keys():
                    centroid_vector = ht_category_mean[category]
                    euclidean_distance = distance.euclidean(centroid_vector,desc2vec)
                    if euclidean_distance < min_distance:
                        category_label = category
                        min_distance = euclidean_distance
                assigned_label_list.append({'vec':desc2vec,'category': str(category_label.lower().strip())})
            except:
                print("Exception occurred for index",index)
    ul_vector_set = pd.DataFrame(assigned_label_list)
    pickle.dump(ul_vector_set,open(aggregated_ul_pickle+"aggregated_ul_data_"+str(min_index)+"_"+str(max_index)+".pkl","wb"))

n = 64
index_list = []
number_of_data,dimension = desc2vec_embeddings.shape
number_of_elements = math.ceil(number_of_data/n)
for i in range(0,n):
    min_index = number_of_elements*i
    max_index = min_index+number_of_elements
    index_list.append({'min_index':min_index,'max_index':max_index})
Parallel(n_jobs=n)(delayed(assign_label)(index["min_index"],index["max_index"]) for index in index_list)



