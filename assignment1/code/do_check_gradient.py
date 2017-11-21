from __future__ import print_function
import os
import sys
#VERSION = 2 if '2.7' in sys.version else 3
import numpy as np
import csv
import zipfile

"""
    You are expected to upload a e*******.zip file, inside the zip file contains 6 gradients.csv file.
    For verification, you can comment out the line11 to 15, and change the truth_path and ID to 'b', however, do note the grading process WILL contain these lines.
"""
# ID = 'e012345678'
truth_path = '/Users/mengjiunchiou/Google Drive/NUS/Modules/1718S1/CS5242 Neural Networks and Deep Learning/assignment/assignment1/Question2_4/b' #change truth_path = 'b' for verification
# zip_ref = zipfile.ZipFile(ID+'.zip', 'r')
# zip_ref.extractall('.')
# zip_ref.close()
file_name = ['dw-100-40-4.csv', 'db-100-40-4.csv', 'dw-28-6-4.csv', 'db-28-6-4.csv', 'dw-14-28-4.csv', 'db-14-28-4.csv'] 
true_file = ['true-dw-100-40-4.csv', 'true-db-100-40-4.csv', 'true-dw-28-6-4.csv', 'true-db-28-6-4.csv', 'true-dw-14-28-4.csv', 'true-db-14-28-4.csv']
threshold = 0.05

def read_file(name):
    l = list()
    with open(name) as f:
        reader = csv.reader(f)
        for row in reader:
            l.append(row)
    return l

"""
    You can try your grading function, while the function is yet to decided.
    However, in the the ideal situation you should expect dis = 0
"""
def do_some_grading(l0, l1, th):
    dis = np.mean(np.abs(l0-l1).astype(float)/(0.1+l1))
    if dis <= th:
        return 1
    else:
        return 0

"""
    The threshold is introduced to address the numerial bias due to rounded floats,
    which could be as small as zero
"""
def compare(sub, true, threshold=0):
    scores = list()
    if not len(sub)==len(true):
        return 0
    for i in range(len(sub)):
        l0 = np.array(sub[i]).astype(np.float)
        l1 = np.array(true[i]).astype(np.float)
        if not len(l0)==len(l1):
            return 0
        else:
            scores.append(do_some_grading(l0, l1, threshold))
    return scores

true_grads = list()
for f in true_file:
    true_grads.append(read_file(os.path.join(truth_path,f)))

score = list()
for i, fn in enumerate(file_name):
    grads = read_file(os.path.join(truth_path,fn))
    s = compare(grads, true_grads[i], threshold)
    score += s
print(np.sum(score))
            
