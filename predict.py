from stanfordcorenlp import StanfordCoreNLP
import json
from collections import Counter

UNK = "<UNKNOWN>"

nlp = StanfordCoreNLP(r"./stanford-corenlp-full-2018-02-27")
allsentences = []

allTagCounts = Counter(json.loads(open("allTagCounts.json","r").readline()))
emissionDists = Counter(json.loads(open("emissionDists.json","r").readline()))
perWordTagCounts = Counter(json.loads(open("perWordTagCounts.json","r").readline()))
transitionDists = Counter(json.loads(open("transitionDists.json","r").readline()))

cnt = 0

for tag in emissionDists:
    emissionDists[tag] = Counter(emissionDists[tag])

with open("test_v2.txt","r") as f:
    for line in f:
        
        thisid, sentence = line.split(",",1)
        if thisid=="id":
            continue
        sentence = sentence.strip()[1:-1]
        
        if not(sentence):
            continue
        
        tagged = nlp.pos_tag(sentence)
        missingwordposition = 0
        leastprob = float("inf")
     
        previousword = tagged[0][0]
        previoustag = tagged[0][1]
        
        
        for idx, each_pair in enumerate(tagged):
            word = each_pair[0]
            tag = each_pair[1]
            
            if idx==0:
                continue
                
            if tag in transitionDists[previoustag]:
                thisprob = transitionDists[previoustag][tag] # * emissionDists[tag][word]
            else:
                thisprob = transitionDists[previoustag][UNK]
            
            if thisprob<leastprob:
                leastprob = thisprob
                missingwordposition = idx
                
        previoustag = tagged[missingwordposition-1][1]
        nexttag = tagged[missingwordposition][1]
        missingtag = ''
        mostprob = float("-inf")
        
        for tag in emissionDists:
            if tag in transitionDists[previoustag]:
                thisprob = transitionDists[previoustag][tag]
            else:
                thisprob = transitionDists[previoustag][UNK]
                
            if nexttag in transitionDists[tag]:
                thisprob += transitionDists[tag][nexttag]
            else:
                thisprob += transitionDists[tag][UNK]
            
            if thisprob > mostprob:
                mostprob = thisprob
                missingtag = tag
                
        missingword = emissionDists[missingtag].most_common(1)[0][0]
        
        splitsentence = sentence.split(" ")
        fullsentence = splitsentence[0:missingwordposition] + ["("+missingword+")"] + splitsentence[missingwordposition:] 
        allsentences.append(" ".join(fullsentence))
        
        cnt+=1
        if cnt%1000==0:
            print(cnt)
            print(" ".join(fullsentence))
        
with open("predict.txt","w") as f:
    for each_sentence in allsentences:
        f.write(each_sentence+"\n")
        
                
        
            
        
        