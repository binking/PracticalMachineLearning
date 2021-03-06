from sklearn.ensemple import RandomForestClassifier
from numpy import genformtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[32001:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
    
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)

    savetxt('submissionRF.csv', rf.predict(test), delimiter=',', fmt='%f')

if __name__=='__main__':
    main()
