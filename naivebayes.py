import math
import argparse
import numpy as np
# Arguments format as shared in assignment
parser = argparse.ArgumentParser()
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)

args = vars(parser.parse_args())
Xtrain_name = args['f1']
Xtest_name = args['f2']
Out_name = args['o']

def compute_outlier_word(word_cnts):
# computing the outlier words based on the frequency
    ol=[]
    counts=list(word_cnts.values())
    mean=np.mean(counts)
    sd=np.std(counts)
    hv=mean+2*sd
    lv=mean-2*sd
    for word in word_cnts.keys():
        count=word_cnts[word]
        if count>hv or count<lv:
            ol.append(word)
    #print mean,sd
    return ol

total_mails=0
dictionary=set()
spamHamDict={"spam":{},"ham":{}}
mail_type={"spam":0,"ham":0}
spam_ham_word_count={"spam":0,"ham":0}
prior_prob={"spam":0.0,"ham":0.0}
reader = open(Xtrain_name, 'r')
cond_prob_ham={}
cond_prob_spam={}
word_count={}
#Reading the training data and computing the Word counts
for line in reader:
    words = line.split(" ")[2:]
    for (index, word) in enumerate(words):
        if index % 2 == 0:
            words += [word]

    for item in words:
        dictionary.add(item)
        if item in word_count:
            word_count[item]+=1
        else:
            word_count[item]=1
reader.close()

outliers=compute_outlier_word(word_count)

for word in dictionary:
    spamHamDict["spam"][word]=0
    spamHamDict["ham"][word] = 0


reader = open(Xtrain_name, 'r')

for line in reader:
    total_mails += 1
    tokens = line.split(" ")
    type = tokens[1]
    mail_type[type] += 1
    for i in range(2, len(tokens), 2):
        spamHamDict[type][tokens[i]] += 1
        spam_ham_word_count[type]+=1

#computing the prior probabailities for spam and ham:
prior_prob["spam"]=float(mail_type["spam"]/float(total_mails))
prior_prob["ham"]=float(mail_type["ham"]/float(total_mails))

# computing the conditional probabaility p(word|spam) and p(word|ham)
for word in dictionary:
    cond_prob_spam[word]=float(spamHamDict["spam"][word]/float(spam_ham_word_count["spam"]))
    cond_prob_ham[word] = float(spamHamDict["ham"][word] / float(spam_ham_word_count["ham"]))


#prediction of labels for the test data
prediction={}

test_reader = open(Xtest_name, 'r')
out_file=open(Out_name,"w")
labels=[]
mail_ids=[]
predictions=[]
for line in test_reader:
    words = line.split(" ")
    prediction_label=""
    mail_id=words[0]
    label=words[1]
    ip_prob={"spam":0.0,"ham":0.0}
    mail_ids.append(mail_id)
    labels.append(label)
    #get actual words
    words=words[2:]
    for (index, word) in enumerate(words):
        if index % 2 == 0 and (word not in outliers):
                words += [word]

    for word in words:
        if word not in dictionary:
            ip_prob["spam"]=ip_prob["spam"]+np.log((1.0/float(spam_ham_word_count["spam"]+2.0))+10**-15)
            ip_prob["ham"] =ip_prob["ham"]+ np.log((1.0/float(spam_ham_word_count["ham"]+2.0))+10**-15)
        else:
            ip_prob["spam"] = ip_prob["spam"] +np.log(float(cond_prob_spam[word])+10**-15)
            ip_prob["ham"] = ip_prob["ham"] +np.log(float(cond_prob_ham[word])+10**-15)

    ip_prob["spam"]=ip_prob["spam"]+np.log(prior_prob["spam"])
    ip_prob["ham"] = ip_prob["ham"] +np.log(prior_prob["ham"])

    if(ip_prob["spam"]>ip_prob["ham"]):
        prediction_label="spam"
    else:
        prediction_label="ham"

    out_file.write(mail_id+" "+prediction_label)
    predictions.append(prediction_label)

test_reader.close()


# computing  accuraccy precision and recall
corr_spam=0
incorr_spam=0
corr_ham=0
incorr_ham=0

for i in range(len(mail_ids)):
    if(predictions[i]==labels[i]):
        if(labels[i]=="spam"):
            corr_spam +=1
        else:
            corr_ham+=1
    else:
        if (labels[i] == "spam"):
            incorr_ham += 1
        else:
            incorr_spam += 1

accuracy=float((corr_ham+corr_spam)/float(len(predictions)))
precision=float(corr_spam/float(corr_spam+incorr_spam))
recall=float(corr_spam/float(corr_spam+incorr_ham))
print("accuracy is :",accuracy ,"precision is :",precision,"recall is:",recall)

