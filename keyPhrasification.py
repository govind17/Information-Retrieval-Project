import pandas
import pke
import pandas as pd
import csv
import json
import gensim


# for each of the best candidates
def key_phrasification():
    # initialize a TopicRank keyphrase extraction model
    extractor = pke.unsupervised.TopicRank()
    rankedDocument_df = pandas.read_csv('RankedDocuments/RankingPrediction.csv')
    trec_docs_df = pandas.read_csv('/Users/GovindShukla/tensorflow-env/trec_docs.csv')

    keys = list(rankedDocument_df.columns.values)
    i1 = trec_docs_df.set_index(keys).index
    i2 = rankedDocument_df.set_index(keys).index
    filtered_docs_df = trec_docs_df[i1.isin(i2)]
    df = pd.DataFrame(columns=['doc_id', 'KeyList'])
    p = 0
    for index, row in filtered_docs_df.iterrows():
        df.loc[p, 'doc_id'] = row['doc_id']
        df.loc[p, 'KeyList'] = getKeys(extractor, row['text'])
        p = p + 1
    # applying merge
    filtered_docs_df = pd.merge(filtered_docs_df, df, on = "doc_id", how = "inner")
    filtered_docs_df.to_csv('RankedDocuments/keyPhrase.csv',index=False)
    csvFilePath = r'RankedDocuments/keyPhrase.csv'
    jsonFilePath = r'RankedDocuments/keyPhrase.json'
    csv_to_json(csvFilePath, jsonFilePath)

def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf:
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf)

        #convert each csv row into python dict
        for row in csvReader:
            #add this python dict to json array
            jsonArray.append(row)

    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

def getKeys(extractor, text):
    extractor.load_document(input=text, language='en')

    # identify the keyphrase candidates using TopicRank's default strategy
    # i.e. the longest sequences of nouns and adjectives `(Noun|Adj)*`
    extractor.candidate_selection()

    # identifying keyphrase candidates populates the extractor.candidates dictionary
    # let's have a look at the keyphrase candidates
    # for each keyphrase candidate
    # In TopicRank, candidate weighting is a three-step process:
    #  1. candidate clustering (grouping keyphrase candidates into topics)
    #  2. graph construction (building a complete-weighted-graph of topics)
    #  3. rank topics (nodes) using a random walk algorithm
    extractor.candidate_weighting()

    # Get the N-best candidates (here, 5) as keyphrases
    keyphrases = extractor.get_n_best(n=5, stemming=False)
    keyphrasesList = []
    for i, (candidate, score) in enumerate(keyphrases):
        keyphrasesList.append(candidate)
        print()
    return keyphrasesList


if __name__ == '__main__':
    key_phrasification()

