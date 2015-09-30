from util.crossValidation import crossValidate
from classifiers import bayes


list256=pickle.load( open( "marchese29/classifier/blob/master/data/256WordList.pickle", "rb" ) )
list1000=pickle.load( open( "marchese29/classifier/blob/master/data/1000WordList.pkl", "rb" ) )

matrix1000=pickle.load( open( "marchese29/classifier/blob/master/data/reducedFeatureVectorMatrix1000.pkl", "rb" ) )
matrix256=pickle.load( open( "marchese29/classifier/blob/master/data/reducedFeatureVectorMatrix256.pkl", "rb" ) )

trainingTestPairs1000=crossValidate(matrix1000)
trainingTestPairs256=crossValidate(matrix256)


accArray1000=NaiveBayesClassifier(trainingTestPairs1000,list1000)
accArray256=NaiveBayesClassifier(trainingTestPairs256,list256)



print("The below line shows 5 accuracies when we use 1000 words.  Each one is for one training/testing fold")
print(accArray1000)

print("The below line shows 5 accuracies when we use 256 words.  Each one is for one training/testing fold")
print(accArray256)



