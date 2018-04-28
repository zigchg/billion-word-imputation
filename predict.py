from stanfordcorenlp import StanfordCoreNLP
import json
from collections import Counter
from math import log

UNK = "<UNKNOWN>"

nlp = StanfordCoreNLP(r"./stanford-corenlp-full-2018-02-27")
allsentences = []

tag2tagweight_tag = .5
word2wordweight_tag = 1

tag2tagweight_word = 1
word2wordweight_word = .5

allTagCounts = Counter(json.loads(open("allTagCounts.json","r").readline()))
emissionDists = Counter(json.loads(open("emissionDists.json","r").readline()))
perWordTagCounts = Counter(json.loads(open("perWordTagCounts.json","r").readline()))
transitionDists = Counter(json.loads(open("transitionDists.json","r").readline()))
word2wordDists = Counter(json.loads(open("word2word.json","r").readline()))

v_size = len(word2wordDists)
alpha = 0.1

unk2any = log((1/v_size+alpha)/(v_size+alpha*v_size))

cnt = 0

for tag in emissionDists:
    emissionDists[tag] = Counter(emissionDists[tag])

lineCounter = 0
    
with open("test_v2.txt","r") as f:
    
    
    for line in f:
        lineCounter+=1
        if lineCounter>100:
            break
        
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
        
        thisprob = 0
        
        for idx, each_pair in enumerate(tagged):
            word = each_pair[0]
            tag = each_pair[1]
            
            if idx==0 or idx==len(tagged)-1:
                continue
                
            if tag in transitionDists[previoustag]:
                thisprob = tag2tagweight_tag * transitionDists[previoustag][tag] # * emissionDists[tag][word]
            else:
                thisprob = tag2tagweight_tag * transitionDists[previoustag][UNK]
            
            if previousword not in word2wordDists:
                thisprob += unk2any
            elif word in word2wordDists[previousword]:
                thisprob += word2wordweight_tag * word2wordDists[previousword][word]
            else:
                thisprob += word2wordweight_tag * word2wordDists[previousword][UNK]
            
            if thisprob<leastprob:
                leastprob = thisprob
                missingwordposition = idx
            
            previousword = word
            previoustag = tag
                
        previoustag = tagged[missingwordposition-1][1]
        nexttag = tagged[missingwordposition][1]
        
        previousword = tagged[missingwordposition-1][0]
        nextword = tagged[missingwordposition][0]
        
        missingtag = ''
        mostprob = float("-inf")
        thisprob = 0
        
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
                
        # missingword = emissionDists[missingtag].most_common(1)[0][0]
        
        mostprob = float("-inf")
        thisprob = 0
        missingword = UNK
        
        for possibleword in emissionDists[missingtag]:
            if possibleword==UNK:
                continue
            
            if possibleword in emissionDists[missingtag]:
                thisprob = tag2tagweight_word * emissionDists[missingtag][possibleword]
            else:
                thisprob = tag2tagweight_word * emissionDists[missingtag][UNK]
            
            if previousword not in word2wordDists:
                thisprob+= unk2any
            elif possibleword in word2wordDists[previousword]:
                thisprob+=word2wordweight_word * word2wordDists[previousword][possibleword]
            else:
                thisprob+=word2wordweight_word * word2wordDists[previousword][UNK]
                
            if possibleword not in word2wordDists:
                thisprob += unk2any
            elif nextword in word2wordDists[possibleword]:
                thisprob += word2wordweight_word * word2wordDists[possibleword][nextword]
            else:
                thisprob += word2wordweight_word * word2wordDists[possibleword][UNK]
            
            if thisprob>mostprob:
                mostprob = thisprob
                missingword = possibleword
        
        splitsentence = sentence.split(" ")
        fullsentence = splitsentence[0:missingwordposition] + ["("+missingword+")"] + splitsentence[missingwordposition:] 
        allsentences.append(" ".join(fullsentence))
        
        cnt+=1
        if cnt%100==0:
            print(cnt)
            print(" ".join(fullsentence))
        
with open("predict.txt","w") as f:
    for each_sentence in allsentences:
        f.write(each_sentence+"\n")
        
                
        
            
        
        