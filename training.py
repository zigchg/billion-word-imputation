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

UNK = "<UNKNOWN>"

# read in statistics
with open("train_v2.txt") as f:
    for line in f:
        try:
            tagged_sentence = nlp.pos_tag(line)
        except:
            print(line)
            continue
        previousTag = ""
        for idx, each_pair in enumerate(tagged_sentence):
            word = each_pair[0]
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
            
            if idx!=1:
                if previousTag not in transitionCounts:
                    transitionCounts[previousTag] = Counter()
                    transitionCounts[previousTag][tag] = 0
                elif tag not in transitionCounts[previousTag]:
                    transitionCounts[previousTag][tag] = 0
                transitionCounts[previousTag][tag] += 1
            
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

nlp.close()

with open("allTagCounts.json") as f:
    f.write(json.dumps(allTagCounts))

with open("perWordTagCounts.json") as f:
    f.write(json.dumps(perWordTagCounts))
    
with open("transitionDists.json") as f:
    f.write(json.dumps(transitionDists))
    
with open("emissionDists.json") as f:
    f.write(json.dumps(emissionDists))