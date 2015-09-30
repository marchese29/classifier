import random


#data is a list of lists
#implements 5-fold cross validation

#returns 5 tuples.  Each tuple contains a training and test data set.
#run your classifier on each of the five training/test pairs
def crossValidate(data):
    
    nonNullTopicDocs=list()
    
    #find all of the docs with non Null topics
    for doc in data:
        #if doc[1][0]!='NULL':
        nonNullTopicDocs.append(doc)

    
    
    FivePairs=list()
    
    size=len(nonNullTopicDocs)
    #print(size)

    subsetSize=size/5

    partition1=subsetSize
    partition2=subsetSize*2
    partition3=subsetSize*3
    partition4=subsetSize*4
    partition5=size



    group1=list()
    group2=list()
    group3=list()
    group4=list()
    group5=list()
    
    #put each doc into one of five equal groups
    for doc in nonNullTopicDocs:
        rand=random.randint(0,size)
        if 0 <= rand <= partition1:
            group1.append(doc)
        elif partition1 <= rand <= partition2:
            group2.append(doc)
        elif partition2 <= rand <= partition3:
            group3.append(doc)
        elif partition3 <= rand <= partition4:
            group4.append(doc)
        else:
            group5.append(doc)

    dataSets=[group1,group2,group3,group4,group5]

    '''select one test data set.  The others are training data.  This is done 5 times, where each set serves as test once'''
    for testData in dataSets:
        training=list()
        test=testData
        for trainingData in dataSets:
            if trainingData!=testData:
                for record in trainingData:
                    training.append(record)
        tuple=[training,test]
        FivePairs.append(tuple)

    return FivePairs
