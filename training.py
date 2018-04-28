from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
from math import log
import json

nlp = StanfordCoreNLP(r"./stanford-corenlp-full-2018-02-27")

ALPHA = .1
allTagCounts = {}
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because missing Counter entries default to 0, not log(0)
transitionDists = {}
emissionDists = {}
word2word = {}
word2wordDists = {}
#tag2tag = {}
#tag2tagDists = {}

UNK = "<UNKNOWN>"

lineCount = 0

# read in statistics
with open("train_v2.txt") as f:
    for line in f:
        lineCount += 1
#        if lineCount > 2000:
#            break
        
        if lineCount%10000==0:
            print(lineCount)
        
        tagged_sentence = nlp.pos_tag(line)
        previousTag = ""
        previousWord = ""
        for idx, each_pair in enumerate(tagged_sentence):
            word = each_pair[0].lower()
            tag = each_pair[1]
            if tag not in allTagCounts:
                allTagCounts[tag] = 0
            allTagCounts[tag] += 1
            
            if word not in perWordTagCounts:
                perWordTagCounts[word] = {}
                perWordTagCounts[word][tag] = 0
            elif tag not in perWordTagCounts[word]:
                perWordTagCounts[word][tag] = 0
            perWordTagCounts[word][tag] += 1
            
            if idx!=0:
                if previousTag not in transitionCounts:
                    transitionCounts[previousTag] = Counter()
                    transitionCounts[previousTag][tag] = 0
                elif tag not in transitionCounts[previousTag]:
                    transitionCounts[previousTag][tag] = 0
                transitionCounts[previousTag][tag] += 1
                
                if previousWord not in word2word:
                    word2word[previousWord] = {}
                    word2word[previousWord][word] = 0
                elif word not in word2word[previousWord]:
                    word2word[previousWord][word] = 0
                word2word[previousWord][word] += 1
                    
#                if previousTag not in tag2tag:
#                    tag2tag[previousTag] = {}
#                    tag2tag[previousTag][tag] = 0
#                elif tag not in tag2tag[previousTag]:
#                    tag2tag[previousTag][tag] = 0
#                tag2tag[previousTag][tag] += 1    
                    
            previousWord = word
            previousTag = tag
            
            if tag not in emissionCounts:
                emissionCounts[tag] = Counter()
                emissionCounts[tag][word] = 0
            elif word not in emissionCounts[tag]:
                emissionCounts[tag][word] = 0
            emissionCounts[tag][word] += 1
                
    # normalize counts and store log probability distributions in transitionDists and emissionDists
    # ...
    
    
    for given_tag in transitionCounts:
        transitionCounts[given_tag][UNK] = 0
        total = sum(transitionCounts[given_tag].values(),0.0)
        length = len(transitionCounts[given_tag])
        transitionDists[given_tag] = {}
        for tag in transitionCounts[given_tag]:
            transitionDists[given_tag][tag] = log((transitionCounts[given_tag][tag]+ALPHA)/(total+ALPHA*length))
            
    for given_tag in emissionCounts:
        emissionCounts[given_tag][UNK] = 0
        total = sum(emissionCounts[given_tag].values(),0.0)
        length = len(emissionCounts[given_tag])
        emissionDists[given_tag] = {}
        for word in emissionCounts[given_tag]:
            emissionDists[given_tag][word] = log((emissionCounts[given_tag][word]+ALPHA)/(total+ALPHA*length))
            
    for given_word in word2word:
        word2word[given_word][UNK] = 0
        total = sum(word2word[given_word].values(),0.0)
        length = len(word2word[given_word])
        word2wordDists[given_word] = {}
        for word in word2word[given_word]:
            word2wordDists[given_word][word] = log((word2word[given_word][word]+ALPHA)/(total+ALPHA*length))
            
#    for given_tag in tag2tag:
#        tag2tag[given_tag][UNK] = 0
#        total = sum(tag2tag[given_tag].values(),0.0)
#        length = len(tag2tag[given_tag])
#        tag2tagDists[given_tag] = {}
#        for tag in tag2tag[given_tag]:
#            tag2tagDists[given_tag][tag] = log((tag2tag[given_tag][tag]+ALPHA)/(total+ALPHA*length))

nlp.close()

with open("allTagCounts.json","w") as f:
    f.write(json.dumps(allTagCounts))

with open("perWordTagCounts.json","w") as f:
    f.write(json.dumps(perWordTagCounts))
    
with open("transitionDists.json","w") as f:
    f.write(json.dumps(transitionDists))
    
with open("emissionDists.json","w") as f:
    f.write(json.dumps(emissionDists))
    
with open("word2word.json","w") as f:
    f.write(json.dumps(word2wordDists))

#with open("tag2tag.json","w") as f:
#    f.write(json.dumps(tag2tagDists))
