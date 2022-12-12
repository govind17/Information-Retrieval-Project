import flask
import import_ipynb
import pandas as pd
from flask import request
from flask_cors import CORS
from rank_bm25 import BM25Plus
from keyPhrasification import key_phrasification

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


# @app.before_first_request
# def _declareStuff():
#     global corpus
#     corpus_df = pd.read_csv(
#         '/Users/GovindShukla/tensorflow-env/trec_docs.csv')
#     corpus = corpus_df['text'].values


# def bm25okapi_search(tokenized_query, corpus, n_results=1):
#     """
#     Function that takes a tokenized query and prints the first 100 words of the
#     n_results most relevant results found in the corpus, based on the BM25
#     method.
#
#     Parameters
#     ----------
#     @param tokenized_query: list, array-like
#         A valid list containing the tokenized query.
#     @param bm25: BM25 object,
#         A valid object of type BM25 (BM25Okapi or BM25Plus) from the library
#         `rank-bm25`, initialized with a valid corpus.
#     @param corpus: list, array-like
#         A valid list containing the corpus from which the BM25 object has been
#         initialized. As returned from function read_corpus().
#     @param n_results: int, default = 1
#         The number of top results to print.
#     """
#
#     # We skip checking validity of arguments for now... We assume the user
#     # knows what they're doing.
#     # Tokenize the corpus
#     tokenized_corpus = [doc.split(" ") for doc in corpus]
#     # Instantiate BM25 object from the tokenized corpus
#     print('Still tokenizing...')
#     bm25 = BM25Plus(tokenized_corpus)
#
#     print('Still ranking...')
#
#     ranked = bm25.get_top_n(tokenized_query, corpus, n=n_results)
#     print('Ranking done...')
#     return ranked


@app.route('/query', methods=['GET'])
def search():
    # corpus_df = pd.read_csv(
    #     '/Users/GovindShukla/Desktop/Information-Retrieval-Project/RankedDocuments/trec_docs_sample.csv')
    # corpus = corpus_df['text'].values
    query = request.args.get('searchString')
    print(query)
    searchResults = pd.read_csv('/Users/GovindShukla/Desktop/Information-Retrieval-Project/RankedDocuments/RankingPrediction.csv')
    searchResultswithkeys = key_phrasification(searchResults)
    return searchResultswithkeys.to_json(orient='records')


@app.route('/feedback', methods=['POST'])
def fetchFeedback():
    # corpus_df = pd.read_csv(
    #     '/Users/GovindShukla/Desktop/Information-Retrieval-Project/RankedDocuments/trec_docs_sample.csv')
    # corpus = corpus_df['text'].values
    list = request.args.get('feedbackList')
    print(request.body)
    print(list)
    return


# if __name__ == "__main__":
#     corpus_df = pd.read_csv(
#         '/Users/GovindShukla/Desktop/Information-Retrieval-Project/RankedDocuments/trec_docs_sample.csv')
#     corpus = corpus_df['text'].values
if __name__ == "__main__":
    app.run(debug=True, port=5000)
