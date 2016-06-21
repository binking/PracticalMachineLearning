# Using graphlab Create
import graphlab as gl
import numpy as np
import os, time

def loadData(csv_file, nrows=0):
    if os.path.exists(csv_file):
        dataset = gl.SFrame.read_csv(csv_file, nrows=nrows, delimiter=',')
    else:
        print "%s is missed" % csv_file
        return -1
    return dataset

def normalize(sframe, skip_cols = 0):
    aver_arr = []; stderror_arr = []
    for col in sframe.column_names()[skip_cols:]:
        aver = sframe[col].mean(); aver_arr.append(aver)
        stderror = sframe[col].std(); stderror_arr.append(stderror)
        sframe[col] = (sframe[col] - aver + 1.0) * 1.0 /  (stderror + 1)
    return sframe, aver_arr, stderror_arr

def main():
    # Loading data
    trainset = loadData('train.csv', 28000); testset = loadData('test.csv', 28000)
    print "Load data finished  .. "
    # Preprocessing train data
    train_data, val_data = trainset.random_split(.8, seed=0)  # seed = 0 and = 1 are different ???
    pixel_feat = train_data.column_names()[1:]
    train_data, aver_arr, std_arr = normalize(train_data, skip_cols=1)
    #print aver_arr; print std_arr; print pixel_feat
    #Preprocessing validation set using the same aver and std error as train data
    for i in range(len(pixel_feat)):
        col = pixel_feat[i]
        val_data[col] = (val_data[col] - aver_arr[i] + 1.0)*1.0 / (std_arr[i]+1) # Because the sparsity, We add 1 for prevent 0
        #print sum(val_data[col])
    for i in range(len(pixel_feat)):
        col = pixel_feat[i]
        testset[col] = (testset[col] - aver_arr[i] + 1.0)*1.0 / (std_arr[i]+1) # Because the sparsity, We add 1 for prevent 0
    print "Preprocessing validation set finished .. "
    # Estimating model
    data_recognizer_model = gl.logistic_classifier.create(train_data, target='label', features=pixel_feat, validation_set=val_data,  max_iterations=30)
    predicted_res = data_recognizer_model.predict(testset) # predict() outputs SArray
    predicted_res.save('result_gl.csv', format='csv')

if __name__=="__main__":
    init_time = time.time()
    main()
    print time.time() - init_time
