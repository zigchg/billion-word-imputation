from stanfordcorenlp import StanfordCoreNLP
import json
from collections import Counter
from math import log

UNK = "<UNKNOWN>"

nlp = StanfordCoreNLP(r"./stanford-corenlp-full-2018-02-27")
allsentences = []

tag2tagweight_blank = .5
word2wordweight_blank = 1

tag2wordweight_word = 1
word2wordweight_word = .5

outputfile = "predict_"+str(tag2tagweight_blank).strip("0.")+"_"+str(word2wordweight_blank).strip("0.")+"_"+str(tag2wordweight_word).strip("0.")+"_"+str(word2wordweight_word).strip("0.")+".csv"

allTagCounts = Counter(json.loads(open("allTagCounts.json","r").readline()))
emissionDists = Counter(json.loads(open("emissionDists.json","r").readline()))
perWordTagCounts = Counter(json.loads(open("perWordTagCounts.json","r").readline()))
transitionDists = Counter(json.loads(open("transitionDists.json","r").readline()))
word2wordDists = Counter(json.loads(open("word2word.json","r").readline()))

v_size = len(word2wordDists)
alpha = 0.1

unk2any = log((1/v_size+alpha)/(v_size+alpha*v_size))

cnt = 0
    
with open("test_v2.txt","r") as f:
    
    for line in f:
        
        # if predict for the wole test dataset, please comment the following two lines out
        if cnt>100:
            break
            
        thisid, sentence = line.split(",",1)
        if thisid.strip('"')=="id":
            continue
        original_sentence = sentence.strip()[1:-1]
        sentence = sentence.strip()[1:-1].lower().replace('""','"')
        
        if not(sentence):
            allsentences.append("")
            continue
        
        tagged = nlp.pos_tag(sentence)
        missingwordposition = 0
        leastprob = float("inf")
     
        previousword = tagged[0][0]
        previoustag = tagged[0][1]
        
        thisprob = 0
        
        # finding the blanks
        for idx, each_pair in enumerate(tagged):
            word = each_pair[0]
            tag = each_pair[1]
            
            if idx==0 or idx==len(tagged)-1:
                continue
                
            if tag in transitionDists[previoustag]:
                thisprob = tag2tagweight_blank * transitionDists[previoustag][tag] # * emissionDists[tag][word]
            else:
                thisprob = tag2tagweight_blank * transitionDists[previoustag][UNK]
            
            if previousword not in word2wordDists:
                thisprob += unk2any
            elif word in word2wordDists[previousword]:
                thisprob += word2wordweight_blank * word2wordDists[previousword][word]
            else:
                thisprob += word2wordweight_blank * word2wordDists[previousword][UNK]
            
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
        
        # identifying POS tag for the missing word
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
                
        
        # selecting actual word based on the POS tag
        mostprob = float("-inf")
        thisprob = 0
        missingword = UNK
        
        for possibleword in emissionDists[missingtag]:
            if possibleword==UNK:
                continue
            
            if possibleword in emissionDists[missingtag]:
                thisprob = tag2wordweight_word * emissionDists[missingtag][possibleword]
            else:
                thisprob = tag2wordweight_word * emissionDists[missingtag][UNK]
            
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
        
        splitsentence = original_sentence.split(" ")
        fullsentence = splitsentence[0:missingwordposition] + [missingword] + splitsentence[missingwordposition:] 
        samplesentence = splitsentence[0:missingwordposition] + ["("+missingword+")"] + splitsentence[missingwordposition:] 
        allsentences.append(" ".join(fullsentence))
        
        cnt += 1
        # output sample sentence per 10 sentences
        # if predicting for whole test dataset, change 10 to 1000
        if cnt%10==0:
            print(cnt)
            print(" ".join(samplesentence))
        
with open(outputfile,"w") as f:
    f.write('"id","sentence"\n')
    idx = 0
    for each_sentence in allsentences:
        idx+=1
        f.write(str("idx")+',"'+each_sentence+'"\n')
        
                
        
            
        
        