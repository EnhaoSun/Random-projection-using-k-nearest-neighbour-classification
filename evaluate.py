'''
 Example usage: python evaluate.py predicted_label.csv test_label.csv
'''
print (__doc__)

import os, sys

def compare_results(predicted_label_file, true_label_file):
    with open(predicted_label_file) as f:
        predicted_labels = f.readlines()
    with open(true_label_file) as f:
        true_labels = f.readlines()
    
    assert(len(predicted_labels) == 100 and len(true_labels) == 100)
    
    error_count = 0
    for i in range(100):
        if (int(predicted_labels[i]) != int(true_labels[i])):
            error_count+=1
    return error_count/float(100)

if __name__ == "__main__":
    assert(len(sys.argv) == 3)
    print "Prediction error:",compare_results(sys.argv[1], sys.argv[2])

