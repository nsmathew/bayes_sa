#!/usr/bin/env python
# coding: utf-8

# In[247]:


#!/usr/bin/env python
import re, random, math, collections, itertools
from matplotlib import pyplot as plt

PRINT_ERRORS=1


# In[137]:


#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = re.findall(r"[a-z\-]+", posDictionary.read())
    # ACR22NSM - loads 'a+'' from dict as 'a' due to regex above. So correcting it to a+
    posWordList.remove('a')
    posWordList.append('a+')
    # ACR22NSM - End
    
    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = re.findall(r"[a-z\-]+", negDictionary.read())

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    #create Training and Test Datsets:
    #We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

  #create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    #create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#


# In[3]:


#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    #iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: #calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                #keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1# keeps count of total words in negative class
                
                #keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        #do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------


# In[278]:


#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):
    pNeg=1-pPos

    #These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    #for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words
        #predictPower={} # ACR22NSM for priting the likelihood probability for words to explain in Q6
        pPosW=pPos
        pNegW=pNeg

        for word in wordList: #calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]
                #predictPower[word]= "Pos:" + str(round(pWordPos[word],5)) + " / Neg:" + str(round(pWordNeg[word],5))
        #print("\nP(W|Class):", predictPower,"\n")
        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
    
    evaluatePerformance(dataName, total, correct, correctpos, correctneg, 
                          totalpospred, totalnegpred, totalpos, totalneg)

# ACR22NSM - Function to calculate the metrics and hence evaluate performance
# 1. Accuracy = # of correctly classified texts / # texts
# 2. Precision_Pos = # texts *correctly* classified as positive / # of texts classified as positive
# 3. Recall_Pos = # texts *correctly* classified as positive / # of *actual* positive texts
# 4. F1 Score = (2 * Precision_Pos * Recall_Pos) / (Precision_Pos + Recall_Pos)
# 5, 6, 7 Similar to 2, 3 and 4 but for the Negative class
def evaluatePerformance(dataName, total, correct, correctpos, 
                         correctneg, totalpospred, totalnegpred, totalpos, totalneg, th=None):
    print("\n-----\t\PERFORMANCE METRIC FOR", dataName ,"-----")
    # Accuracy
    if total == 0:
        accuracy = 0
    else:
        accuracy = correct / total * 100
    print("Overall Accuracy %0.2f%%" % accuracy)
    
    # Precision for Positive class
    if totalpospred == 0:
        precisionPos = 0
    else:    
        precisionPos = correctpos / totalpospred * 100
    print("Precision for Positive class %d/%d - %0.2f%%" % (correctpos, totalpospred, precisionPos))

    # Recall for Positive class
    if totalpos == 0:
        recallPos = 0
    else:    
        recallPos = correctpos / totalpos * 100
    print("Recall for Positive class %d/%d - %0.2f%%" % (correctpos, totalpos, recallPos))
    
    # F1 Score for Positive class
    if precisionPos == 0 and recallPos == 0:
        f1Pos = 0
    else:        
        f1Pos = (2 * precisionPos * recallPos)/(precisionPos + recallPos)
    print("F1 score for Positive class %0.2f%%" % f1Pos)

    # Precision for Negative class
    if totalnegpred == 0:
        precisionNeg = 0
    else:        
        precisionNeg = correctneg / totalnegpred * 100
    print("Precision for Negative class %d/%d - %0.2f%%" % (correctneg, totalnegpred, precisionNeg))
    
    # Recall for Negative class
    if totalneg == 0:
        recallNeg = 0
    else:     
        recallNeg = correctneg / totalneg * 100
    print("Recall for Negative class %d/%d - %0.2f%%" % (correctneg, totalneg, recallNeg))
    
    # F1 Score for Negative class
    if precisionNeg == 0 and recallNeg == 0:
        f1Neg = 0
    else:          
        f1Neg = (2 * precisionNeg * recallNeg)/(precisionNeg + recallNeg)
    print("F1 score for Negative class %0.2f%%\n" % f1Neg)
    plotMetrics(dataName, accuracy, precisionPos, recallPos, f1Pos, precisionNeg, recallNeg, f1Neg, th)

# ACR22NSM - Utiltity function to plot a graph of performance metrics
def plotMetrics(dataName, accuracy, precisionPos, recallPos, f1Pos, precisionNeg, recallNeg, f1Neg, th=None):
    #figure(figsize=(4,4))
    plt.plot(["Acc.", "Pos Prec.", "Pos Recall", "Pos F1", "Neg Prec.", "Neg Recall", "Neg F1"],
             [accuracy, precisionPos, recallPos, f1Pos, precisionNeg, recallNeg, f1Neg], 'bo-')
    plt.title(dataName.replace("\t",""), fontsize=14)
    plt.grid(True)
    plt.ylabel("Percentage", fontsize=14)
    plt.ylim(0, 110)
    plt.xlabel('Metrics', fontsize=14)
    plt.annotate(str(round(accuracy,1))+"%", (0, accuracy), xytext=(-10, 10), textcoords='offset points')
    plt.annotate(str(round(precisionPos,1))+"%", (1, precisionPos), color="green", xytext=(-10, 10), textcoords='offset points')
    plt.annotate(str(round(recallPos,1))+"%", (2, recallPos), color="green", xytext=(-10, 10), textcoords='offset points')
    plt.annotate(str(round(f1Pos,1))+"%", (3, f1Pos), color="green", xytext=(-10, 10), textcoords='offset points')
    plt.annotate(str(round(precisionNeg,1))+"%", (4, precisionNeg), color="red", xytext=(-10, 10), textcoords='offset points')
    plt.annotate(str(round(recallNeg,1))+"%", (5, recallNeg), color="red", xytext=(-10, 10), textcoords='offset points')
    plt.annotate(str(round(f1Neg,1))+"%", (6, f1Neg), color="red", xytext=(-10, 10), textcoords='offset points')
    plt.annotate("Threshold: "+str(th), (5, 10), color="black")
    plt.show()


# In[264]:


# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
# ACR22NSM - Change function signature to also include neighbourRange for additional rules application
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold, neighbourRange=-1):
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        # ACR22NSM - Additional weight to determine if sign invert should be applied
        weight=0
        matchedSentWords = set(Words).intersection(set(sentimentDictionary.keys()))
        #print("SENTIMENT WORDS:", matchedSentWords)
        for matched in matchedSentWords:
            weight += sentimentDictionary[matched]
        #for word in Words: #ACR22NSM changed to use enumerate to get an index
        for idx, word in enumerate(Words):
            if word in sentimentDictionary:
                score+=sentimentDictionary[word]
        # ACR22NSM - Adjust scoring using additional rules START
                if neighbourRange > 0: # Toggle for additional rules execution, should be > 0. default is -1
                    # Pick out the neighbouring words backward by 'neighbour_range'
                    neighbours = set()
                    for i in range(-1*neighbourRange, 0, 1):
                        if i == 0 or idx+i < 0 or idx+i >= len(Words): #Exx
                            continue                    
                        neighbours.add(Words[idx+i])
                    # Run additional rules for the word to adjust scoring
                    score = additionalRulesEngine(word, sentimentDictionary[word], weight, neighbours, score)
        #Additional score adjustment for exclamation
        if neighbourRange > 0:
            score = additionalRulesEngine(word='!', wordSentiScore=0, weight=0, neighbours=None, score=score) if '!' in sentence else score
        # ACR22NSM - Adjust scoring using additional rules END
        
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %score + sentence)
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %score + sentence)

    evaluatePerformance(dataName, total, correct, correctpos, correctneg, 
                          totalpospred, totalnegpred, totalpos, totalneg, threshold)


# In[263]:


# ACR22NSM - Additional function to populate the intensifier, diminisher words and the weights for them
def populateReferences(rulesTokenDict, rulesScoreDict):
    # Input data
    # Data loaded directly in code for simplicity. Can be replace with a lexicon as well
    negations = set([
                    "no","not","never","none","nobody","nor","nothing","nowhere", "cannot", #Negative words
                    "doesn't","wasn't","didn't","can't","isn't","shouldn't","won't" #Contracted forms
                    ]) 

    diminishers = set(["few","hardly","little","rarely","scarcely","seldom", "slightly", "somewhat",
                      "minor", "colosally", "extremely", "particularly"])
    diminisherWeight = 0.5
    intensifiers = set(["so","extremely","very","too","rather","really","absolutely",
                    "completely","highly","totally","utterly", "especially", "fully", "supremely"])
    intesifierWeight = 1
    
    exclamationWeight = 0.5
    
    # Load into a dictionary
    rulesTokenDict['negations'] = negations
    rulesTokenDict['diminishers'] = diminishers
    rulesTokenDict['intensifiers'] = intensifiers

    rulesScoreDict['diminishers'] = diminisherWeight
    rulesScoreDict['intensifiers'] = intesifierWeight
    rulesScoreDict['exclamation'] = exclamationWeight
    
# ACR22NSM - Rules to adjust the score for a sentence based on diminisher, intensifers and negation
# in the n-gram neighbourhood. Also adjust score based on presence of exclamation mark
def additionalRulesEngine(word=None, wordSentiScore=0, weight=0, neighbours=None, score=0):
    global rulesTokenDict, rulesScoreDict
    # Negation Rule
    negations = set(rulesTokenDict.get('negations', None))
    matchedNegations = neighbours.intersection(negations) if neighbours is not None else None
    if matchedNegations is not None and len(matchedNegations) == 1 and score != 0:
        score -= wordSentiScore
        #if weight <=1 and weight >=-1: # Enable this block if sign invert is required
            #score = -1 * score
    # Diminisher Rule
    diminishers = set(rulesTokenDict.get('diminishers', None))
    matchedDiminishers = neighbours.intersection(diminishers) if neighbours is not None else None
    if matchedDiminishers is not None and len(matchedDiminishers) > 0:
        if wordSentiScore <0:
            score += rulesScoreDict['diminishers']
        else:
            score -= rulesScoreDict['diminishers']
    # Intesifier Rule
    intensifiers = set(rulesTokenDict.get('intensifiers', None))
    matchedIntensifiers = neighbours.intersection(intensifiers) if neighbours is not None else None
    if matchedIntensifiers is not None and len(matchedIntensifiers) > 0:
        if wordSentiScore <0:
            score -= rulesScoreDict['intensifiers']
        else:
            score += rulesScoreDict['intensifiers']
    # Excalamation Rule    
    if word == '!':
        if score <0:
            score -= rulesScoreDict['exclamation']
        else:
            score += rulesScoreDict['exclamation']
        
    return score


# In[7]:


#Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)
    print(pWordPos['the'], pWordNeg['the'], pWord['the'])
    print(pWordPos['unfunny'], pWordNeg['unfunny'], pWord['unfunny'])
    print(pWordPos['exquisitely'], pWordNeg['exquisitely'], pWord['exquisitely'])
    
    return sortedPower

# Additional function to check how of the top sentiment words from the training are
# in the sentiment dictionary
def checkWordInSentimentDictinary(sentimentDictionary, sortedPower, pWord, n_lst):
    results_pos = []
    results_neg = []
    unavailable_pos = []
    unavailable_neg = []
    for n in n_lst:
        head, tail = set(sortedPower[:n]), set(sortedPower[len(sortedPower)-n:])
        sentiDict = set(list(sentimentDictionary.keys()))
        print("\nAvailability of top sentiment words in sentiment dictionary")
        
        unavailable_pos.append(tail - sentiDict)
        posCnt = n - len(tail - sentiDict)
        posPct = 100 * posCnt / n

        unavailable_neg.append(head - sentiDict)
        negCnt = n - len(head - sentiDict)
        negPct = 100 * negCnt / n
        print("For %d top words" % n)
        print("TOP POSITIVE WORDS - %d available (%0.2f%%)" % (posCnt, posPct))
        print("TOP NEGATIVE WORDS - %d available (%0.2f%%)" % (negCnt, negPct))
        print("\nPOSITIVE Words not found in Sentiment Dictionary: ", unavailable_pos)
        print("\nNEGATIVE Words not found in Sentiment Dictionary: ", unavailable_neg)
        results_pos.append(posPct)
        results_neg.append(negPct)
    
    x = range(0,len(n_lst))
    plt.bar([i-0.2 for i in x], results_pos, width=0.4, label = 'Positive Words')
    plt.bar([i+0.2 for i in x], results_neg, width=0.4, label = 'Negative Words')
    plt.xticks(x, n_lst)
    plt.ylabel("Percentage", fontsize=14)
    plt.xlabel('Top n Words', fontsize=14)
    plt.title("How many most useful predictors are in\n the Sentiment Dictionary", fontsize=14)
    plt.legend()
    for i in x:
        plt.text(i-0.2, results_pos[i], str(results_pos[i])+"%", ha = 'center')
        plt.text(i+0.2, results_neg[i], str(results_neg[i])+"%", ha = 'center')
    plt.show()
    


# In[279]:


### ---------- Main Script --------------------------
random.seed(1784401)

sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

# ACR22NSM - AdditionalRules dictionaries START
rulesTokenDict = {}
rulesScoreDict = {}
populateReferences(rulesTokenDict, rulesScoreDict)
# ACR22NSM - AdditionalRules dictionaries END

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

#build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

#run naive bayes classifier on datasets
print ("Naive Bayes")
#testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5) # Uncommneted for Step 3, point 1
#testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7) # Uncommneted for Step 3, point 1


#run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
#testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
#testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)

# ACR22NSM - Additional function calls with additional rules enabled
#testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1, 4)
#testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1, 4)
#testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1, 4)

# print most useful words
#sortedPower = mostUseful(pWordPos, pWordNeg, pWord, 100)
# ACR22NSM - Additional function to check if high prediction power words are in the dictionary
#checkWordInSentimentDictinary(sentimentDictionary, sortedPower, pWord, [100, 250, 500])

