import pyterrier as pt

if not pt.started():
    pt.init()

import os

cord19 = pt.datasets.get_dataset('irds:cord19/trec-covid')
pt_index_path = './terrier_cord19'
if not os.path.exists(pt_index_path + "/data.properties"):
    # create the index, using the IterDictIndexer indexer
    indexer = pt.index.IterDictIndexer(pt_index_path)
    # we give the dataset get_corpus_iter() directly to the indexer
    # while specifying the fields to index and the metadata to record
    index_ref = indexer.index(cord19.get_corpus_iter(),
                              fields=('abstract',),
                              meta=('docno',))
else:
    # if you already have the index, use it.
    index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")
    index = pt.IndexFactory.of(index_ref)
