import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
import spacy
import re
#agginto da Mirok per fare in modo che il path funzioni si su Windows che su Linux
import os
import pickle
from spacy.tokens import DocBin

#DECISIONTREE
from sklearn.datasets import load_iris
from sklearn import tree
#naive
from sklearn.naive_bayes import GaussianNB

doc_bin = DocBin(store_user_data=True)


nlp = spacy.load(r"C:\Windows\System32\.env\Lib\site-packages\it_core_news_lg\it_core_news_lg-3.4.0")
#nlp = spacy.load("it_core_news_lg")
nlp.max_length=4000000
"""
prende in input una lista di token (spacy) e restiuisce il numero di femminili individuati
"""
def contaFem(tokens):
    contfem = 0
    for token in tokens:
     gender = str(token.morph.get("Gender"))
     verbi = str(token.pos_)
     if (gender == "['Fem']"):
         if (verbi == "AUX" or verbi == "VERB" or verbi == "ADJ"):
            contfem = contfem + 1
    return contfem
"""
prende in input una lista di token (spacy) e restiuisce il numero di femminili individuati
"""
def contaMasc(tokens):
    contmasc = 0
    for token in tokens:
     gender = str(token.morph.get("Gender"))
     verbi = str(token.pos_)
     if (gender == "['Masc']"):
         if (verbi == "AUX" or verbi == "VERB" or verbi == "ADJ"):
            contmasc = contmasc + 1
    return contmasc
"""
prende in input un testo e restiuisce il numero di parole che terminano con a o e
"""
def matchea(testo):
    regola = "([a-zA-Z]{1,}[ae])[^a-z-A-Z]"  # parole che finiscono per 'a'  oppure 'e'
    match = re.findall(regola, testo)
    return len(match)

#with open(r"C:\Users\matti\Desktop\Tesi\final_package_train_test\final_package_train_test\training.txt", "r", encoding="utf8") as f:
#	file = f.read()
print("Caricamento dati di training...")
f = open("TAG-it_train_test"+os.sep+"final_package_train_test"+os.sep+"training.txt", "r", encoding="utf-8")
file = f.read()

soup = BeautifulSoup(file, "lxml")
users = soup.find_all('user')

#users=users[0:15] #in fase di test del codice non ho bisogno di caricare tutto, ma mi basta solo una parte di utenti per provare il codice

train= [""] * len(users)
for i in range(0,len(users)):
    print("riga ",i,"in elaborazione su ",len(users))
    posts=users[i].find_all("post")
    for j in range(0,len(posts)):
        train[i]= train[i] +' '+ posts[j].get_text()
train=np.array(train)

print("Estrazione label..")
label_train=[""]*len(users)
cont=0
for i in range(0,len(users)):
    label_train[i]=users[i].get("gender")
label_train=np.array(label_train)

print("Estrazione pos tagging... ci sarà molto da attendere...")
#valutare di salvare il pos tagging in un file, in modo da non rifare l'elaborazione tutte le volte

if os.path.isfile("serialized docs.pickle"):
    file=open("serialized docs.pickle", "rb")
    bytes_data=pickle.load(file)
    file.close()
    doc_bin = DocBin().from_bytes(bytes_data)
    train_tokens = list(doc_bin.get_docs(nlp.vocab))
else:
    train_tokens= []
    for i in range(0,len(users)):
        print("riga ",i,"in elaborazione su ",len(users))
        posts=users[i].find_all("post")
        text=''
        for j in range(0,len(posts)):
            text+=' '+posts[j].get_text()
        user_tokens = nlp(text)
        train_tokens.append(user_tokens)
        doc_bin.add(user_tokens)
    bytes_data = doc_bin.to_bytes()
    file=open("serialized docs.pickle", "wb")
    pickle.dump(bytes_data, file)
    file.close()


#print(train_tokens[0])

print("Calcolo BoW")
vectorizer_word = CountVectorizer(analyzer='word', ngram_range=(1, 6))
vectorizer_word = vectorizer_word.fit(train)
X_word_vectorize = vectorizer_word.transform(train)

print("Calcolo BoPoS")
vectorizer_pos = CountVectorizer(analyzer='word', ngram_range=(1, 6))

"""
train_tokens e un vettore che ha un entry per ogni utente
user è il placeholder per l'utente ed è un vettore di tutti i token (spacy) dei suoi post concatenati
"""

train_tokens_to_text=[]
for user in train_tokens:
    pos_text=''
    for token in user:
        pos_text+=' '+token.pos_
    train_tokens_to_text.append(pos_text)
#print(train_tokens_to_text[0]) #per capire qual è l'output
vectorizer_pos = vectorizer_pos.fit(train_tokens_to_text)
X_pos_vectorize = vectorizer_pos.transform(train_tokens_to_text)
#print("dizionario:", vectorizer_pos.get_feature_names()) #se vuole vedere il dizionario

print("Calcolo lenght")
X_lenght=[]
for i in range(0,len(users)):
    X_lenght.append([matchea(train[i]), contaFem(train_tokens[i]), contaMasc(train_tokens[i]), len(train[i]), len(train[i].split()), len(train[i]) / len(train[i].split())])
X_lenght = np.array(X_lenght)

"""
Quindi creo un array numpy per ogni gruppo di feature prima della kfold.
Uso gli  indici dati da kf.split per separare tra training e test dei diversi gruppi di feature
"""

print("Avvio k-fold validation")
kf = KFold(n_splits=5)
kf.get_n_splits(train)
fmacros=[0]*5
cont=0
for train_index, test_index in kf.split(train):
    print("Fold",cont+1)

    #prima divido train e test della Bow
    X=X_word_vectorize[train_index]
    Y=X_word_vectorize[test_index]

    #poi "stacko" le BoW con le BoPoS
    #X = hstack((X, X_pos_vectorize[train_index]))
    #X=X_pos_vectorize[train_index]
    #Y=X_pos_vectorize[test_index]
    #Y = hstack((Y, X_pos_vectorize[test_index]))

    #qui "stacko" anche il gruppo di feature sulle lunghezze
    X = hstack((X, X_lenght[train_index]))
    Y = hstack((Y, X_lenght[test_index]))

    print("(righe, colonne):", X.shape)
    """
    clf = SVC(kernel="linear")
    clf.fit(X, label_train[train_index])

    pred = clf.predict(Y)

    #decision tree

    clf = tree.DecisionTreeClassifier()
    clf=clf.fit(X,label_train[train_index])

    pred=clf.predict(Y)
    """
    
    #naive
    
    clf = GaussianNB()
    clf.fit(X.toarray(), label_train[train_index])
    pred=clf.predict(Y.toarray())

    fmacros[cont]=f1_score(label_train[test_index],pred,average="macro")
    cont+=1


print("f-Macro Media",np.average(fmacros))
