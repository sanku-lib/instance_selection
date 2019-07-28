import pickle
import csv
import sys
import re
import sys
import os
import csv
from joblib import Parallel, delayed
import boto.ec2
import time
import traceback



# specify AWS keys
auth = {"aws_access_key_id": "XXXXXXXXXXXX", "aws_secret_access_key": "XXXXXXXXXXXXX"}
instance_id = "XXXXXXXXXX"
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


def dump_dict_as_csv(dict_object,file_path):
    print(str(file_path.name))
    dump_file = open(str(file_path.name),"w")
    for key in dict_object:
        try:
            dump_file.write(str(key))
            dump_file.write(',')
            dump_file.write(str(dict_object[key]))
            dump_file.write("\n")
        except Exception as e:
            print("exception in dump as csv : ",str(e))
    dump_file.close()

kds_data_src = sys.argv[1]
population_data_src = sys.argv[2]
vocabulary_dest = sys.argv[3]


#Read knowledge driven sample data and build vocabulary
base_vocabulary = {}
with open(kds_data_src) as data_file:
    csv_reader = csv.reader(data_file)
    for row in csv_reader:
        description = str(row[0]).strip().lower()
        words = description.split()
        for word in words:
            if word in base_vocabulary:
                base_vocabulary[word] = base_vocabulary[word] + 1
            else:
                base_vocabulary[word] = 1
print("base vocabulary build completed")
#Dump base vocabulary and auxiliary vocabulary
print("Size of base vocabulary: ",len(base_vocabulary.keys()))
try:
    pickle.dump(base_vocabulary,open(os.path.join(vocabulary_dest,'base_vocabulary.pkl'),'wb'))
    dump_dict_as_csv(base_vocabulary,open(os.path.join(vocabulary_dest,'base_vocabulary.csv')))
except Exception as e:
    print("exception in dumping vocabulary. ")
    print("Exception Details: ",str(e))


#Read population data and add description which are not present in base_vocabulary
auxiliary_vocabulary= {}
for file_name in os.listdir(population_data_src):
    with open(os.path.join(population_data_src,file_name)) as data_file:
        csv_reader = csv.reader(data_file)
        for row in csv_reader:
            try:
                description = str(row[0]).strip().lower()
                words = description.split()
                for word in words:
                    if word in base_vocabulary:
                        base_vocabulary[word] += 1
                    else:
                        if word in auxiliary_vocabulary:
                            auxiliary_vocabulary[word] += 1
                        else:
                            auxiliary_vocabulary[word] = 1
            except Exception as e:
                print("exception occurred for ",row)
                print("exception details: ",str(e))
print('completed auxiliary vocabulary')

print("size of auxiliary vocabulary: ",len(auxiliary_vocabulary.keys()))
#Dump base vocabulary and auxiliary vocabulary
try:
    pickle.dump(auxiliary_vocabulary,open(os.path.join(vocabulary_dest,'auxiliary_vocabulary.pkl'),'wb'))
    dump_dict_as_csv(auxiliary_vocabulary,open(os.path.join(vocabulary_dest,'auxiliary_vocabulary.csv')))
except Exception as e:
    print("exception to dump auxiliary vocabulary")
    print(str(e))
print("dumped base and auxiliary vocabulary")
time.sleep(5)
stopInstance()
