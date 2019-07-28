# Proprocess Unlabeled data: takenize descriptions, calculate word distribution, estimation score of descriptions and sort descriptions based on score
# Author: Shibsankar Das
# Usage: python3 preprocess_ul_data.py <path to ul data> <path to dump tokenized,sorted description> <path to dump sorted word distribution based on frequency>

import boto.ec2
import pickle
import sys
import csv
import os
import re
from joblib import Parallel, delayed


# specify AWS keys
auth = {"aws_access_key_id": "XXXXXXXXXXXXX", "aws_secret_access_key": "XXXXXXXXXXXXXXXXXX"}
instance_id = "XXXXXXXXXXXXX"
# Set csv file pool to system's maximum limit
csv.field_size_limit(sys.maxsize)

def stopInstance():
    print("Stopping the instance...")
    try:
        ec2 = boto.ec2.connect_to_region("us-west-2", **auth)
    except:
        sys.exit(0)
    try:
         ec2.stop_instances(instance_ids=instance_id)
    except:
        sys.exit(0)



ul_data_src = sys.argv[1]
clean_ul_data_dest = sys.argv[2]
destination_prob_pkl = sys.argv[3]


def tokenize(x):
    tk_x = x

    # list of characters which needs to be replaced with space
    space_replace_chars = [':', ',', '"', '[', ']', '~', '*', ';', '!', '?', '(', ')', '@', '&']
    tk_x = tk_x.translate({ord(x): ' ' for x in space_replace_chars})

    # list of characters which needs to be removed
    # remove_chars = ['-',"'",'.']
    remove_chars = ["'", '.']  # keeping - as it is for now
    tk_x = tk_x.translate({ord(x): '' for x in remove_chars})

    # replace all consecutive spaces with one space
    tk_x = re.sub('\s+', ' ', tk_x).strip()

    # find all consecutive numbers present in the word, first converted numbers to * to prevent conflicts while replacing with numbers
    regex = re.compile(r'([\d])')
    tk_x = regex.sub('*', tk_x)
    nos = re.findall(r'([\*]+)', tk_x)

    # replace the numbers with the corresponding count like 123 by 3
    for no in nos:
        tk_x = tk_x.replace(no, str(len(no)), 1)

    return tk_x

# Calculate frequency of words in the corpus
word_list = {}
for file_name in os.listdir(ul_data_src):
    with open(os.path.join(ul_data_src,file_name)) as data_file:
        csv_reader = csv.reader(data_file)
        for row in csv_reader:
            description = tokenize(str(row[0]).strip().lower())
            words = description.split()
            for word in words:
                if word in word_list:
                    word_list[word]+=1
                else:
                    word_list[word]=1

print("Calculated frequency of all words")
#Dump vocabulary with frequency of word
pickle.dump(word_list,open(os.path.join(destination_prob_pkl,"ul_word_vocabulary.pkl"),"wb"))
# Sort words based on frequency,dump and clear dict
sorted_word_on_freq = [(key,word_list[key]) for key in sorted(word_list,key=word_list.get,reverse=True)]
pickle.dump(sorted_word_on_freq,open(os.path.join(destination_prob_pkl,"word_prob_distribution.pkl"),"wb"))
sorted_word_on_freq.clear()

# Calculate total number of unique words in the corpus
total_words = sum(word_list.values())

# Estimate probability of words in the corpus
word_prob_distribution = {}
for word in word_list:
    word_prob_distribution[word] = word_list[word]/total_words

# Calculate score of description
description_score_list = {}
for file_name in os.listdir(ul_data_src):
    with open(os.path.join(ul_data_src,file_name)) as data_file:
        csv_reader = csv.reader(data_file)
        for row in csv_reader:
            description = tokenize(str(row[0]).strip().lower())
            if description not in description_score_list:
                words = description.split()
                score = 0
                number_of_words = 0
                if len(words) == 0:
                    print("info: empty description")
                else:
                    for word in words:
                        number_of_words += 1
                        if word in word_prob_distribution:
                            score += (1-word_prob_distribution[word])
                    score /= number_of_words
                    description_score_list[description] = score

print("Calculated score of all descriptions")
sorted_description_on_score = [(description) for description in sorted(description_score_list,key=description_score_list.get,reverse=True)]
print("Sorted clean description based on score")
pickle.dump(sorted_description_on_score,open(os.path.join(clean_ul_data_dest,"sorted_unlabeled_data.pkl"),"wb"))
print("Dumped sorted description into pickle")
print("Execution completed successfully")
