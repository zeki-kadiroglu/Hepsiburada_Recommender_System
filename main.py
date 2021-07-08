# Importing necessary libraries
import json

import gensim
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect
import pickle
import random
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
#from wtforms import SelectField
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
#from flask_login import current_user,login_manager,LoginManager,UserMixin

with open(r"C:\Users\MONSTER\Desktop\reco\rec_final.pkl", 'rb') as file:
    rec_model = pickle.load(file)


df = pd.read_csv(r'C:\Users\MONSTER\Desktop\reco\cleaned_data.csv')


customers = df["sessionid"].unique().tolist()

# shuffle customer ID's
random.shuffle(customers)

# extract  customer ID's
customers_train = [customers[i] for i in range(len(customers))]

# split data into train and validation set
train_df = df[df['sessionid'].isin(customers_train)]


# list to capture purchase history of the customers
purchases_train = []


products = train_df[["productid", "clean_name"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='productid', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('productid')['clean_name'].apply(list).to_dict()

def cleaning(data):
    import re
    stop_words = nltk.corpus.stopwords.words('turkish')
    # lem = WordNetLemmatizer()

    # büyük türkçe karakter olduğu için text olarak alamıyoruz,bu yüzden stadart hale getirdim
    line = re.sub(r"[İ]", 'i', data)
    line2 = re.sub(r"[Ş]", 'ş', line)
    line3 = re.sub(r"[Ü]", 'ü', line2)
    line4 = re.sub(r"[Ö]", 'ö', line3)
    line5 = re.sub(r"[Ç]", 'ç', line4)

    # save some words
    text = re.sub(r"[-+():;.',!?]", '', line5.lower())

    # text = re.sub("-()'{}<>[]\/&%+^!?:;", '', data.lower())
    # 1 . Tokenize
    text_tokens = word_tokenize(text)

    # 2. Remove Puncs
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]

    # 3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    # 4. lemma
    # text_cleaned = [lem.lemmatize(t) for t in tokens_without_sw]
    # join
    return ' '.join(tokens_without_sw)
#
#
def del_noi(row): # en sonda kalan ekleri silecek öncesinde her satır sonuna bir boşluk eklendi " "
    remove_lst = ["gr", "g", "ml", "cc", "l", "lt", "x cm", "lı", "li", "lu", "lü"
              ,'kg','x','cm','adet']
    new_row = re.sub(r"(?<=\s)gr(?=\s)", "", row)
    new2 = re.sub(r"(?<=\s)g(?=\s)", "", new_row)
    new3 = re.sub(r"(?<=\s)ml(?=\s)", "", new2)
    new4 = re.sub(r"(?<=\s)cc(?=\s)", "", new3)
    new5 = re.sub(r"(?<=\s)l(?=\s)", "", new4)
    new6 = re.sub(r"(?<=\s)lt(?=\s)", "", new5)
    new7 = re.sub(r"(?<=\s)x cm(?=\s)", "", new6)
    new8 = re.sub(r"(?<=\s)lı(?=\s)", "", new7)
    new9 = re.sub(r"(?<=\s)li(?=\s)", "", new8)
    new10 = re.sub(r"(?<=\s)lu(?=\s)", "", new9)
    new11 = re.sub(r"(?<=\s)lü(?=\s)", "", new10)
    new12 = re.sub(r"(?<=\s)litre(?=\s)", "", new11)
    new13 = re.sub(r"(?<=\s)kg(?=\s)", "", new12)
    new14 = re.sub(r"(?<=\s)mt(?=\s)", "", new13)
    new15 = re.sub(r"(?<=\s)adet(?=\s)", "", new14)


    return new15

def add_space(item):
    return item +' '

def remove_space(item):
    return item.rstrip()





app = Flask(__name__)





@app.route('/test', methods=['GET'])
def welcome():
    return {"test":"welcome to Hepsiburada"}




@app.route('/pick_product', methods=['GET'])
def pick_product():

    products_list = {"ZYHPDROETTTL022":"Dr.Oetker Poğaça 252 Gr",
                     "HBV00000NVZGU":"Dana Biftek 250 gr",
                     "HBV00000NVZBY":"Domates Salkım 500 gr",
                     "HBV00000PVCIJ":"Tarım Kredi İnce Uzun Makarna 500 g",
                     "ZYCANN63125":"Canbebe Bebek Bakım Örtüsü 10'lu",
                     "HBV00000PVALH":"Off Aerosol Sinekkovar 100 ml",
                     "HBV00000PV7A2":"Eti Form Kepekli zeytinli Kraker 28 g",
                     "HBV00000PLGXG":"Activia Shot Ahududu & Hibiskus 80 Ml",
                     "HBV00000SP6ZG":"Şampuan Men Cool Sport Mentol 600 ml",
                     "ZYSULKORO29243":"Koroplast 100 Mt Streç Film Kesme Bıçağı Hediye"
                     }

    return jsonify(products_list)







@app.route('/recommendation/<item>', methods=['GET','POST'])
def similar_products(item, n=10+1):
    clean = cleaning(item)
    add_spaces = add_space(clean)
    del_noisy = del_noi(add_spaces)
    removed = remove_space(del_noisy)
    if request.method=="POST":
        for i in products_dict:
            if products_dict[i][0] == removed:
                removed = rec_model[str(i)]

        # extract most similar products for the input vector
        ms = rec_model.similar_by_vector(removed, topn=n + 1)[1:]

        # extract name and similarity score of the similar products
        new_ms = []
        new_ms2 = []
        for j in ms:
            pair = (products_dict[j[0]][0], j[1])
            idx = df[df.clean_name==pair[0]].index[0]
            rec_product = df.loc[idx, 'name']

            new_ms.append(rec_product)
            new_ms2.append(pair[1])

    new_items = dict(zip(new_ms, new_ms2))

    return jsonify(sorted(new_items.items(), key=lambda x:x[1],reverse = True))





if __name__ == '__main__':
    app.run(host="0.0.0.0")

