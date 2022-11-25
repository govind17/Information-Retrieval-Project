from types import MethodDescriptorType
import flask
from flask import jsonify, request
from flask import json
from flask_cors import CORS
import pandas as pd

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


def bm25okapi_search(tokenized_query, bm25, corpus, n_results=1):
    """
    Function that takes a tokenized query and prints the first 100 words of the
    n_results most relevant results found in the corpus, based on the BM25
    method.

    Parameters
    ----------
    @param tokenized_query: list, array-like
        A valid list containing the tokenized query.
    @param bm25: BM25 object,
        A valid object of type BM25 (BM25Okapi or BM25Plus) from the library
        `rank-bm25`, initialized with a valid corpus.
    @param corpus: list, array-like
        A valid list containing the corpus from which the BM25 object has been
        initialized. As returned from function read_corpus().
    @param n_results: int, default = 1
        The number of top results to print.
    """

    # We skip checking validity of arguments for now... We assume the user
    # knows what they're doing.

    # Get top results for the query
    return bm25.get_top_n(tokenized_query, corpus, n=n_results)


@app.route('/heroes', methods=['GET'])
def heroes():
    corpus_df = pd.read_csv('/Users/GovindShukla/Desktop/Information-Retrieval-Project/RankedDocuments/trec_docs_sample.csv')
    corpus = corpus_df['text'].values
    query = request.args.get('searchString')
    tokenized_query = query.split(" ")
    searchResults = bm25okapi_search(tokenized_query, corpus, 10)
    print(searchResults)
    return jsonify(searchResults)

app.run()

