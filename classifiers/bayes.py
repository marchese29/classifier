
import copy
import math
import operator





#return the classes in the training set and their respective counts
def createClassMap(trainingSet):
    uniqueClasses={}
    for index in trainingSet:
        firstClass=index[1][0]
        if firstClass not in uniqueClasses:
            uniqueClasses[firstClass]=1
        else:
            uniqueClasses[firstClass]=uniqueClasses[firstClass]+1
    return uniqueClasses





#class_map contains all classes from docs and their counts
#words is all the words we are counting
#returns a list of lists.  each list contains a word, and a map where key is class and value is count
def createMeanDataStructure(class_map,words):
    
    conditionalCount={}
    for index in class_map:
        conditionalCount[index]=0
    
    struct={}
    for word in words:
        map=copy.deepcopy(conditionalCount)
        struct[word]=map


    return struct







#empty meanStruct
#the feature vector
#returns a map.  key is each word.  value is map of average counts for each class.
def fillMeanStruct(meanStruct,data,class_map):
  
  #for each class, find the total number of times the word appears in all documents of that class.
  #meanStruct is each word, with class counts for each class
    for doc in data:
        classLabel=doc[1][0]
        
        for word in doc[2]:
            meanStruct[word][classLabel]=meanStruct[word][classLabel]+doc[2][word]

#tranform counts in meanStruct to means
    for word, classCounts in meanStruct.iteritems():
        #word is the word
        #classCounts is the map of counts for each class associated with a given word
        for cla in class_map:
            #cla is the class name
            classCounts[cla]=(classCounts[cla])/float(class_map[cla])

    return meanStruct

#filledMeanStruct=fillMeanStruct(meanStruct,docs[1:2000],class_map)



#returns a map containing each class and its prior probability
def createPriorStruct(class_map,data):

    totalDocs=len(data)
    class_priors=copy.deepcopy(class_map)

    for clas in class_priors:
        class_priors[clas]=class_priors[clas]/float(totalDocs)

    return class_priors




#computes poisson probability for the observation with given lambda
def poissonProbability(obs,lam):
    
        if obs>170:
            print('greater than 170')
            obs=170
        
        rightNum=math.exp(-lam)
        leftNum=math.pow(lam,obs)
        denom=math.factorial(obs)
        numerator=rightNum*leftNum
        result=(numerator)/denom

        return result



#takes  document from the data and calculates numerator of bayes theorom given classA
def probabilityForClass(document,classA,meanStruct,class_priors):

    bayesNum=1
    for word in document[2]:
        count=document[2][word]
        lam=meanStruct[word][classA]
        indivProb=poissonProbability(count,lam)
        indivProb=indivProb+.01
        bayesNum=bayesNum*indivProb

    prior=class_priors[classA]

    bayesNum=bayesNum*prior

    return bayesNum

#return a map of all the classes and their bayesNums for the given document
def probabilitiesForAllClasses(document,meanStruct,class_priors):

    #allBayesNums is a map; class is value, key is Bayes Numerator
    allBayesNums={}
    for className in class_priors:
        bayesVal=probabilityForClass(document,className,meanStruct,class_priors)
        allBayesNums[className]=bayesVal

    return allBayesNums



#estimates lambda's with trainingData. Uses lambdas to classify on testData.Returns accuracy of classifier on test
def runClassifier(trainingData,testData,selectedWords):
    
    classMap=createClassMap(trainingData)
    
    meanStruct=createMeanDataStructure(classMap,selectedWords)
    meanStruct=fillMeanStruct(meanStruct,trainingData,classMap)
    
    classPriors=createPriorStruct(classMap,trainingData)
    #offline cost is above
    
    #online cost is below
    num=0
    correct=0
    totalPredictions=float(len(testData))
    #testResults=list()
    for doc in testData:
        bayesNums=probabilitiesForAllClasses(doc,meanStruct,classPriors)
        
        #the following line determines the class associated the max bayes value
        likelyClass=max(bayesNums.iteritems(),key=operator.itemgetter(1))[0]
        
        #testResults.append([doc[0],doc[1][0],bayesNums,likelyClass])
        num=num+1
        print (num)
        if (likelyClass==doc[1][0]):
            correct=correct+1

    accuracy=correct/totalPredictions
    return accuracy


#Naive Bayes classifier run on all training/testing pairs.  Returns array of five accuracies on each test set
def NaiveBayesClassifier(data,selectedWords):
    
    accuracyArray=[]
    
    for pair in data:
        accuracy=runClassifier(pair[0],pair[1],selectedWords)
        accuracyArray.append(accuracy)
    
    return accuracyArray
