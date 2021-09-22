from flask import Flask, jsonify, request
import pandas as pd
import re
import math
from urllib.request import Request, urlopen
import json
import nltk

app = Flask(__name__)

dct = {}
files = ['cs', 'fp', 'g', 'rm', 'ss']

for file in files:
    base = 'https://supermarket-51da2-default-rtdb.firebaseio.com/'
    response = urlopen(Request(base + f'{file}.json'))
    data = json.loads(response.read())
    dct[file] = pd.json_normalize(data)

cs = dct['cs']
fp = dct['fp']
g = dct['g']
rm = dct['rm']
ss = dct['ss']
cs["Product"] = cs["Tag"] + ' ' + cs["Name"]
g["Product"] = g["Brand"] + ' ' + g["Product Name"]
ss = ss[ss['Category'] == 'Alcohol']

def IDF(series, kw):
    # IDF = log_e(Total number of documents / Number of documents with term t in it)
    def check(name):
        name = "".join(re.findall("[a-z]+", str(name).lower()))
        return kw in name
    return math.log(len(series) / (sum(series.apply(check)) + 1e-9))

def SumTFIDF(row, keyword, idf):
    kws = re.findall("[a-z]+", keyword.lower())
    name = "".join(re.findall("[a-z]+", str(row).lower()))
    match = 0
    for kw in kws:
        if kw in name:
            match += idf[kw] / len(name)
    return match


def search(name, series):
    # construct IDF
    idf = {}
    kws = re.findall("[a-z]+", str(name).lower())
    for kw in kws:
        idf[kw] = IDF(series, kw)

    # compute score
    df = pd.DataFrame({'Name': series})
    df['Search'] = df['Name'].apply(lambda x: SumTFIDF(x, name, idf))
    suspect = df.sort_values('Search', ascending=False).iloc[0]

    if suspect['Search'] != 0:
        candidate = suspect['Name']
        if nltk.word_tokenize(name)[0].lower() in candidate.lower():
            return suspect['Name']
        else:
            return None
    else:
        return None

@app.route('/')
def hello_world():  # put application's code here
    keyword = request.args.get('keyword')
    print(keyword)
    json_file = {}
    json_file['ss'] = search(keyword, ss['Name'])
    json_file['cs'] = search(keyword, cs['Product'])
    json_file['g'] = search(keyword, g['Product'])
    json_file['fp'] = search(keyword, fp['Product_Name'])
    json_file['rm'] = search(keyword, rm['Name'])
    return jsonify(json_file)

if __name__ == '__main__':
    app.run(host='10.0.2.2', port=5000)
