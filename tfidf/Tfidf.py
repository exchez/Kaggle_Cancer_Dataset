import pandas as pd
import numpy as np
from collections import Counter
import itertools
from scipy.sparse import csr_matrix

class Tfidf():
    def __init__(self, corpus_tokens):
        self.word_count = [dict(Counter(x[0])) for x in corpus_tokens]
        self.doc_freq = self.make_doc_freq(corpus_tokens)
        self.tfidf_vectors = self.make_tfidf_vecs(self.word_count, self.doc_freq)
        self.corpus_matrix = self.make_corpus_matrix(self.tfidf_vectors)
        
    def make_doc_freq(self, corpus_tokens):
        terms = []
        for doc in corpus_tokens:
            terms.append(list(np.unique(doc[0])))
        terms = list(itertools.chain.from_iterable(terms))
        return dict(Counter(terms))

    def make_tfidf_vecs(self, word_counts, doc_f):
        tfidf_docs = []
        n_docs = len(word_counts)
        for index, doc in enumerate(word_counts):
            tfidf_docs.append(self.make_tfidf(doc, doc_f, n_docs))
        return pd.Series(tfidf_docs)
    
    def make_tfidf(self, doc, doc_f, n_docs):
            return {key:(value*(np.log(n_docs/(1+doc_f[key])))) for key, value in doc.items()}

    def make_corpus_matrix(self, corpus_vector_series):
        stack_array = self.make_stack(corpus_vector_series)
        vocab = list( set(stack_array['word']) )
        vocab_df = pd.DataFrame({'word':vocab})
        vocab_df['index'] = vocab_df.index
        master_df = pd.DataFrame(stack_array).set_index(['word']).join(vocab_df.set_index(['word']))
        i = master_df['doc_index']
        j = master_df['index']
        v = master_df['value']
        width = master_df['doc_index'].max() + 1
        height = master_df['index'].max() + 1
        corpus = csr_matrix((v, (i, j)), shape=(width, height))
        return corpus

    def compute_stack_size(self, series):
        return np.sum(list( map(lambda x: len(x.keys()) , series) ))
    
    def make_stack(self, series):
        stack_size = self.compute_stack_size(series)
        new = np.empty(stack_size, dtype=[('doc_index', np.uint), ('word', "S30"), ('value', np.float)])
        counter = 0
        for row in series.iteritems():
            for word in row[1]:
                new[counter] = (row[0], word, row[1][word])
                counter +=1
        return new
        
    def cosine_distance(self, x, y):
        xy = x.dot(y.T)
        dist = xy/(self.norm(x)*self.norm(y))
        return 1-dist[0,0]

    def norm(self, x):
        sum_sq=x.dot(x.T)
        norm=np.sqrt(sum_sq)
        return(norm)

    def query(self, article_index, size=10): #article_index is the row from the articles df
        article = self.corpus_matrix[article_index,:]
        iterable = ((x, self.cosine_distance(article, self.corpus_matrix[x,:])) for x in range(self.corpus_matrix.shape[0]))
        articles_by_distance = np.fromiter(iterable, dtype='uint,float', count=self.corpus_matrix.shape[0])
        articles_by_distance = pd.DataFrame(articles_by_distance).rename(columns={'f1':'cosine_distance', 'f0':'index'}).sort_values(by='cosine_distance')
        return articles_by_distance[0:size]