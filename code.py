import numpy as np
import csv
import time

np.random.seed(1234)
def randomize(): np.random.seed(time.time())

random_mean = 0
random_std = 0.0030
learningrate = 0.001

def height_run(epoch_count = 10, mb_size = 10, report = 1): #run the train&testing
    load_heightdata()
    init_model() 
    train_test(epoch_count, mb_size, report)


def load_heightdata():
    with open('/Users/taeho/Desktop/project0/GaltonFamilies2.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None) #skip header row
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, inputcount, outputcount
    inputcount, outputcount = 4, 1
    data = np.zeros(len(rows), inputcount + outputcount) #create 2dm table in a size of adjusted input and output data
    
    for n, row in enumerate(rows):
        if row[2] == 'male': #one hot incoding for gender -> male/female as each index 
            data[n, 0] = 1
        else:
            data[n, 1] = 1
        data[n, [2,3,4]] = [row[0], row[1], row[3]]
        data[n, 4] = row[3]

def init_model():
    global weight, bias, inputcount, outputcount
