import nltk
import warnings
warnings.filterwarnings("ignore")
# nltk.download() # for downloading packages
#import tensorflow as tf
import numpy as np
import random
import string # to process standard python strings

#nltk.download('punkt')         # first-time use only
#nltk.download('wordnet')        # first-time use only

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f1=open('corpus1.txt','r',errors = 'ignore')
f2=open('corpus2.txt','r',errors = 'ignore')
f3=open('corpus3.txt','r',errors = 'ignore')
f4=open('corpus4.txt','r',errors = 'ignore')
f5=open('corpus5.txt','r',errors = 'ignore')
f6=open('corpus6.txt','r',errors = 'ignore')
f7=open('corpus7.txt','r',errors = 'ignore')
f8=open('corpus8.txt','r',errors = 'ignore')
f9=open('corpus9.txt','r',errors = 'ignore')
f10=open('corpus10.txt','r',errors = 'ignore')
f11=open('corpus11.txt','r',errors = 'ignore')
f12=open('corpus12.txt','r',errors = 'ignore')
f13=open('corpus13.txt','r',errors = 'ignore')
f14=open('corpus14.txt','r',errors = 'ignore')
f15=open('corpus15.txt','r',errors = 'ignore')
f16=open('corpus16.txt','r',errors = 'ignore')
f17=open('corpus17.txt','r',errors = 'ignore')
f18=open('corpus18.txt','r',errors = 'ignore')
f19=open('corpus19.txt','r',errors = 'ignore')
f20=open('corpus20.txt','r',errors = 'ignore')
f21=open('corpus21.txt','r',errors = 'ignore')
f22=open('corpus22.txt','r',errors = 'ignore')
f23=open('corpus23.txt','r',errors = 'ignore')
f24=open('corpus24.txt','r',errors = 'ignore')
checkpoint = "./chatbot_weights.ckpt"

raw1=f1.read()                                      #1. Diabetes 
raw1=raw1.lower()            
sent_tokens1 = nltk.sent_tokenize(raw1)      
word_tokens1 = nltk.word_tokenize(raw1)
sent_tokens1[:2]
word_tokens1[:5]  

raw2=f2.read()                                      #2.Obesity
raw2=raw2.lower()            
sent_tokens2 = nltk.sent_tokenize(raw2)      
word_tokens2 = nltk.word_tokenize(raw2)
sent_tokens2[:2]
word_tokens2[:5]          

raw3=f3.read()                                      #3.High Blood Pressure
raw3=raw3.lower()            
sent_tokens3 = nltk.sent_tokenize(raw3)      
word_tokens3 = nltk.word_tokenize(raw3)
sent_tokens3[:2]
word_tokens3[:5]          

raw4=f4.read()                                      #4.Low Blood Pressure
raw4=raw4.lower()            
sent_tokens4 = nltk.sent_tokenize(raw4)      
word_tokens4 = nltk.word_tokenize(raw4)
sent_tokens4[:2]
word_tokens4[:5]          

raw5=f5.read()                                      #5.Liver Disease
raw5=raw5.lower()            
sent_tokens5 = nltk.sent_tokenize(raw5)      
word_tokens5 = nltk.word_tokenize(raw5)
sent_tokens5[:2]
word_tokens5[:5]     

raw6=f6.read()                                      #6.Heart Disease
raw6=raw6.lower()            
sent_tokens6 = nltk.sent_tokenize(raw6)      
word_tokens6 = nltk.word_tokenize(raw6)
sent_tokens6[:2]
word_tokens6[:5]  

raw7=f7.read()                                      #7.Migrane
raw7=raw7.lower()            
sent_tokens7 = nltk.sent_tokenize(raw7)      
word_tokens7 = nltk.word_tokenize(raw7)
sent_tokens7[:2]
word_tokens7[:5]          

raw8=f8.read()                                    #8.Diarrhoea
raw8=raw8.lower()            
sent_tokens8 = nltk.sent_tokenize(raw8)      
word_tokens8 = nltk.word_tokenize(raw8)
sent_tokens8[:2]
word_tokens8[:5]          

raw9=f9.read()                                    #9.Depression / Anxiety
raw9=raw9.lower()            
sent_tokens9 = nltk.sent_tokenize(raw9)      
word_tokens9 = nltk.word_tokenize(raw9)
sent_tokens9[:2]
word_tokens9[:5]          

raw10=f10.read()                                      #10. Cancer
raw10=raw10.lower()            
sent_tokens10 = nltk.sent_tokenize(raw10)      
word_tokens10 = nltk.word_tokenize(raw10)
sent_tokens10[:2]
word_tokens10[:5] 

raw11=f11.read()                                      #11. Kidney Disease
raw11=raw11.lower()            
sent_tokens11 = nltk.sent_tokenize(raw11)      
word_tokens11 = nltk.word_tokenize(raw11)
sent_tokens11[:2]
word_tokens11[:5]  

raw12=f12.read()                                        #12. Insomnia
raw12=raw12.lower()            
sent_tokens12 = nltk.sent_tokenize(raw12)      
word_tokens12 = nltk.word_tokenize(raw12)
sent_tokens12[:2]
word_tokens12[:5]          

raw13=f13.read()                                            #13. Allergy
raw13=raw13.lower()            
sent_tokens13 = nltk.sent_tokenize(raw13)      
word_tokens13 = nltk.word_tokenize(raw13)
sent_tokens13[:2]
word_tokens13[:5]          

raw14=f14.read()                                            #14.Dengue
raw14=raw14.lower()            
sent_tokens14 = nltk.sent_tokenize(raw14)      
word_tokens14 = nltk.word_tokenize(raw14)
sent_tokens14[:2]
word_tokens14[:5]          

raw15=f15.read()                                            #15.Malaria
raw15=raw15.lower()            
sent_tokens15 = nltk.sent_tokenize(raw15)      
word_tokens15 = nltk.word_tokenize(raw15)
sent_tokens15[:2]
word_tokens15[:5]     

raw16=f16.read()                                              #16.Corona / COVID19
raw16=raw16.lower()            
sent_tokens16 = nltk.sent_tokenize(raw16)      
word_tokens16 = nltk.word_tokenize(raw16)
sent_tokens16[:2]
word_tokens16[:5]  

raw17=f17.read()                                           #17.Influenza/Flu
raw17=raw17.lower()            
sent_tokens17 = nltk.sent_tokenize(raw17)      
word_tokens17 = nltk.word_tokenize(raw17)
sent_tokens17[:2]
word_tokens17[:5]          

raw18=f18.read()                                            #18.Pneumonia
raw18=raw18.lower()            
sent_tokens18 = nltk.sent_tokenize(raw18)      
word_tokens18 = nltk.word_tokenize(raw18)
sent_tokens18[:2]
word_tokens18[:5]          

raw19=f19.read()                                            #19.Asthma
raw19=raw19.lower()            
sent_tokens19 = nltk.sent_tokenize(raw19)      
word_tokens19 = nltk.word_tokenize(raw19)
sent_tokens19[:2]
word_tokens19[:5]          

raw20=f20.read()                                             #20.Osteoarthritis
raw20=raw20.lower()            
sent_tokens20 = nltk.sent_tokenize(raw20)      
word_tokens20 = nltk.word_tokenize(raw20)
sent_tokens20[:2]
word_tokens20[:5] 

raw21=f21.read()                                              #21.Tuberculosis / TB
raw21=raw21.lower()            
sent_tokens21 = nltk.sent_tokenize(raw21)      
word_tokens21 = nltk.word_tokenize(raw21)
sent_tokens21[:2]
word_tokens21[:5]          

raw22=f22.read()                                                #22.Swine flue
raw22=raw22.lower()            
sent_tokens22 = nltk.sent_tokenize(raw22)      
word_tokens22 = nltk.word_tokenize(raw22)
sent_tokens22[:2]
word_tokens22[:5]          

raw23=f23.read()                                                #23.HIV / AIDS
raw23=raw23.lower()            
sent_tokens23 = nltk.sent_tokenize(raw23)      
word_tokens23 = nltk.word_tokenize(raw23)
sent_tokens23[:2]
word_tokens23[:5]          

raw24=f24.read()                                                  #24.Leukemia
raw24=raw24.lower()            
sent_tokens24 = nltk.sent_tokenize(raw24)      
word_tokens24 = nltk.word_tokenize(raw24)
sent_tokens24[:2]
word_tokens24[:5] 

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
Introduce_Ans = ["My name is Remdex.","My name is Remdex and I will answer your queries.","Im Remdex :) ","My name is Remdex and I am happy to solve your queries :) "]
GREETING_INPUTS = ("hello", "hi","hiii","hii","hiiii","hiiii", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hii there", "hi there", "hello", "I am glad! You are talking to me"]
Basic_Q = ("what is m+ store ?","what is m+ store","what is m+ store?","What is m+ store.")
Basic_Ans = "M + Store is an Online Medical Store.We supply medicines at your doorstep. You can order medicines in our website you need by uploading proper prescription.Kindly go through our website once to know better."
Basic_Q1 = ("from where you collect medicine?","from where you collect medicine","from where you collect medicine.","where can I get medicine?","where can I get medicine","where can I get medicine.")
Basic_Ans1 = "We collect generic medicines and supply it to your doorstep at a discount price.We collect it from different authentic sellers and Pradhan Mantri Bhartiya Jan Aushadhi Pariyojana Kendra.The list of Jan Aushadhi Pariyojana Kendras are given in oour website"
Basic_Q2 = ("how much you charge?","how much you charge","how much you charge.","what is the price of medicine?","what is the price of medicine","what is the price of medicine.")
Basic_Ans2 = "We supply generic medicines at a discount price.Please go through our website for more information about medicine price"
Basic_Q3 = ("what is the difference between a brand name and a generic drug?","what is generic medicine?","what is generic medicine","what is generic medicine.")
Basic_Ans3 = "When a medication is first developed, the manufacturer has patent rights on the formula and/or compound. Once this patent right expires, other companies can produce generic versions of the drug that meet the same FDA requirements and regulations as the brand name drug. Most insurance companies require generic substitutions unless specifically requested by the prescriber or patient.We supply generic medicines.It is always advisable to take medicines only after consulting a doctor."

# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Checking for Basic_Q
def basic(sentence):
    for word in Basic_Q:
        if sentence.lower() == word:
            return Basic_Ans

def basic1(sentence):
    for word in Basic_Q1:
        if sentence.lower() == word:
            return Basic_Ans1

def basic2(sentence):
    for word in Basic_Q2:
        if sentence.lower() == word:
            return Basic_Ans2

def basic3(sentence):
    for word in Basic_Q3:
        if sentence.lower() == word:
            return Basic_Ans3

# Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)

SYMPTOM_RESPONSES =["Ohhh..Shit","Sorry to hear that","That doesn't sound good at all","Be cautious about yourself"]
def findsymptom():
  return random.choice(SYMPTOM_RESPONSES)

# Generating response 1
def response1(user_response):
    robo_response=''
    sent_tokens1.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens1)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens1[idx]
        return robo_response

# Generating response 2
def response2(user_response):
    robo_response=''
    sent_tokens2.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens2)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens2[idx]
        return robo_response

# Generating response 3
def response3(user_response):
    robo_response=''
    sent_tokens3.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens3)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens3[idx]
        return robo_response
        
# Generating response 4
def response4(user_response):
    robo_response=''
    sent_tokens4.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens4)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens4[idx]
        return robo_response

# Generating response 5
def response5(user_response):
    robo_response=''
    sent_tokens5.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens5)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens5[idx]
        return robo_response

# Generating response 6
def response6(user_response):
    robo_response=''
    sent_tokens6.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens6)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens6[idx]
        return robo_response

# Generating response 7
def response7(user_response):
    robo_response=''
    sent_tokens7.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens7)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens7[idx]
        return robo_response

# Generating response 8
def response8(user_response):
    robo_response=''
    sent_tokens8.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens8)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens8[idx]
        return robo_response

# Generating response 9
def response9(user_response):
    robo_response=''
    sent_tokens9.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens9)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens9[idx]
        return robo_response

# Generating response 10
def response10(user_response):
    robo_response=''
    sent_tokens10.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens10)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens10[idx]
        return robo_response

# Generating response 11
def response11(user_response):
    robo_response=''
    sent_tokens11.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens11)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens11[idx]
        return robo_response

# Generating response 12
def response12(user_response):
    robo_response=''
    sent_tokens12.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens12)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens12[idx]
        return robo_response

# Generating response 13
def response13(user_response):
    robo_response=''
    sent_tokens13.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens13)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens13[idx]
        return robo_response

# Generating response 14
def response14(user_response):
    robo_response=''
    sent_tokens14.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens14)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens14[idx]
        return robo_response

# Generating response 15
def response15(user_response):
    robo_response=''
    sent_tokens15.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens15)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens15[idx]
        return robo_response

# Generating response 16
def response16(user_response):
    robo_response=''
    sent_tokens16.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens16)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens16[idx]
        return robo_response

# Generating response 17
def response17(user_response):
    robo_response=''
    sent_tokens17.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens17)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens17[idx]
        return robo_response

# Generating response 18
def response18(user_response):
    robo_response=''
    sent_tokens18.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens18)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens18[idx]
        return robo_response

# Generating response 19
def response19(user_response):
    robo_response=''
    sent_tokens19.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens19)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens19[idx]
        return robo_response

# Generating response 20
def response20(user_response):
    robo_response=''
    sent_tokens20.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens20)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens20[idx]
        return robo_response

# Generating response 21
def response21(user_response):
    robo_response=''
    sent_tokens21.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens21)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens21[idx]
        return robo_response

# Generating response 22
def response22(user_response):
    robo_response=''
    sent_tokens22.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens22)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens22[idx]
        return robo_response

# Generating response 23
def response23(user_response):
    robo_response=''
    sent_tokens23.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens23)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens23[idx]
        return robo_response

# Generating response 24
def response24(user_response):
    robo_response=''
    sent_tokens24.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens24)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens24[idx]
        return robo_response

def chat(user_response):
    user_response=user_response.lower()
    keyword = "m+"
    keywordone = "store"
    keywordsecond = "medicine"

    keyS1 = "fever"
    keyS2 = "headache"
    keyS3 = "vomitting"
    keyS4 = "sore throat"
    keyS5 = "cough"
    keyS6 = "breathing problem"
    keyS7 = "pain"
    keyS8 = "itching"

    key0 = "disease"
    key1 = "diabetes"
    key2 = "obesity"
    key3 = "high blood pressure"
    key4 = "low blood pressure"
    key5 = "liver"
    key6 = "heart"
    key7 = "migrane"
    key8 = "diarrhoea"
    key8a = "diarrhea"
    key9 = "depression"
    key9a = "anxiety"
    key10 = "cancer"
    key11 = "kidney"
    key12 = "insomnia"
    key13 = "allergy"
    key14 = "dengue"
    key15 = "malaria"
    key16 = "corona"
    key16a = "covid19"
    key17 = "influenza"
    key17a = "flu"
    key18 = "pneumonia"
    key19 = "asthma"
    key20 = "osteoarthritis"
    key21 = "tuberculosis"
    key21a = "tb"
    key22 = "swine flu"
    key23 = "hiv"
    key23a = "aids"
    key24 = "leukemia"

    exit_list = ['exit','see you later','bye','quit','break']
    if user_response.lower() not in exit_list:
      if(user_response=='thanks' or user_response=='thank you' ):
        flag=False
        #print("ROBO: You are welcome..")
        return "You are welcome.."
      elif(basic(user_response)!=None):
        return basic(user_response)
        #print(basic(user_response))
      elif(basic1(user_response)!=None):
        return basic1(user_response)
        #print(basic1(user_response))
      elif(basic2(user_response)!=None):
        return basic2(user_response)
        #print(basic2(user_response))
      elif(basic3(user_response)!=None):
        return basic3(user_response)
        #print(basic3(user_response))
      elif(user_response.find(keywordsecond ) != -1):
         mans = "Please take any medicine only after consulting with a doctor.You can order medicine from M+ store by uploading a proper prescription. "
         return mans
         #print(mans)
      else:
        if(user_response.find(keyword) != -1 or user_response.find(keywordone) != -1):
          #print("REMDEX: ",end="")
          #print(basic(user_response))
          return basic(user_response)
        elif(user_response.find(keyS1) != -1 or user_response.find(keyS2) != -1 or user_response.find(keyS3) != -1 or user_response.find(keyS4) != -1 or user_response.find(keyS5) != -1 or user_response.find(keyS6) != -1 or user_response.find(keyS7) != -1 or user_response.find(keyS8) != -1):
          sans = "This could be a symptom..Please Consult a Doctor ASAP."
          #print(findsymptom())
          return (findsymptom() +  '\n' + sans)
          #print(sans)
        elif(user_response.find(key1) != -1):                             #1
                #print("REMDEX: ",end="")
                #print(response1(user_response))
                return response1(user_response)
                sent_tokens1.remove(user_response)
        elif(user_response.find(key2) != -1):                             #2
                #print("REMDEX: ",end="")
                #print(response2(user_response))
                return response2(user_response)
                sent_tokens2.remove(user_response)
        elif(user_response.find(key3) != -1):                             #3
                #print("REMDEX: ",end="")
                #print(response3(user_response))
                return response3(user_response)
                sent_tokens3.remove(user_response)
        elif(user_response.find(key4) != -1):                             #4
                #print("REMDEX: ",end="")
                #print(response4(user_response))
                return response4(user_response)
                sent_tokens4.remove(user_response)
        elif(user_response.find(key5) != -1 and user_response.find(key0) != -1):                             #5
                #print("REMDEX: ",end="")
                #print(response5(user_response))
                return response5(user_response)
                sent_tokens5.remove(user_response)
        elif(user_response.find(key6) != -1 and user_response.find(key0) != -1):                             #6
                #print("REMDEX: ",end="")
                #print(response6(user_response))
                return response6(user_response)
                sent_tokens6.remove(user_response)
        elif(user_response.find(key7) != -1):                             #7
                #print("REMDEX: ",end="")
                #print(response7(user_response))
                return response7(user_response)
                sent_tokens7.remove(user_response)
        elif(user_response.find(key8) != -1 or user_response.find(key8a) != -1):                             #8
                #print("REMDEX: ",end="")
                #print(response8(user_response))
                return response8(user_response)
                sent_tokens8.remove(user_response)
        elif(user_response.find(key9) != -1 or user_response.find(key9a) != -1):                             #9
                #print("REMDEX: ",end="")
                #print(response9(user_response))
                return response9(user_response)
                sent_tokens9.remove(user_response)
        elif(user_response.find(key10) != -1):                             #10
                #print("REMDEX: ",end="")
                #print(response10(user_response))
                return response10(user_response)
                sent_tokens10.remove(user_response)
        elif(user_response.find(key11) != -1):                             #11
                #print("REMDEX: ",end="")
                #print(response11(user_response))
                return response11(user_response)
                sent_tokens11.remove(user_response)
        elif(user_response.find(key12) != -1):                             #12
                #print("REMDEX: ",end="")
                #print(response12(user_response))
                return response12(user_response)
                sent_tokens12.remove(user_response)
        elif(user_response.find(key13) != -1):                             #13
                #print("REMDEX: ",end="")
                #print(response13(user_response))
                return response13(user_response)
                sent_tokens13.remove(user_response)
        elif(user_response.find(key14) != -1):                             #14
                #print("REMDEX: ",end="")
                #print(response14(user_response))
                return response14(user_response)
                sent_tokens14.remove(user_response)
        elif(user_response.find(key15) != -1):                             #15
                #print("REMDEX: ",end="")
                #print(response15(user_response))
                return response15(user_response)
                sent_tokens15.remove(user_response)
        elif(user_response.find(key16) != -1 or user_response.find(key16a) != -1):                             #16
                #print("REMDEX: ",end="")
                #print(response16(user_response))
                return response16(user_response)
                sent_tokens16.remove(user_response)
        elif(user_response.find(key17) != -1 or user_response.find(key17a) != -1):                             #17
                #print("REMDEX: ",end="")
                #print(response17(user_response))
                return response17(user_response)
                sent_tokens17.remove(user_response)
        elif(user_response.find(key18) != -1):                             #18
                #print("REMDEX: ",end="")
                #print(response18(user_response))
                return response18(user_response)
                sent_tokens18.remove(user_response)
        elif(user_response.find(key19) != -1):                             #19
                #print("REMDEX: ",end="")
                #print(response19(user_response))
                return response19(user_response)
                sent_tokens19.remove(user_response)
        elif(user_response.find(key20) != -1):                             #20
                #print("REMDEX: ",end="")
                #print(response20(user_response))
                return response20(user_response)
                sent_tokens20.remove(user_response)
        elif(user_response.find(key21) != -1 or user_response.find(key21a) != -1):                             #21
                #print("REMDEX: ",end="")
                #print(response21(user_response))
                return response21(user_response)
                sent_tokens21.remove(user_response)
        elif(user_response.find(key22) != -1):                             #22
                #print("REMDEX: ",end="")
                #print(response22(user_response))
                return response22(user_response)
                sent_tokens22.remove(user_response)
        elif(user_response.find(key23) != -1 or user_response.find(key23a) != -1):                             #23
                #print("REMDEX: ",end="")
                #print(response23(user_response))
                return response23(user_response)
                sent_tokens23.remove(user_response)
        elif(user_response.find(key24) != -1):                             #24
                #print("REMDEX: ",end="")
                #print(response24(user_response))
                return response24(user_response)
                sent_tokens24.remove(user_response)
      
        elif(greeting(user_response)!=None):
          #print("REMDEX: "+greeting(user_response))
          return greeting(user_response)
        elif(user_response.find("your name") != -1 or user_response.find(" your name") != -1 or user_response.find("your name ") != -1 or user_response.find(" your name ") != -1):
          return IntroduceMe(user_response)
          #print(IntroduceMe(user_response))
        
        else:
          #print("REMDEX: ",end="")
          cans = "I am just a chatbot. Please consult a doctor for your query."
          #print(cans)
          return cans
                
    else:
        flag=False
        #print("ROBO: Bye! Take care..Chat with you later!!")
        return "ROBO: Bye! Take care..Chat with you later!!"       
                
        
        
