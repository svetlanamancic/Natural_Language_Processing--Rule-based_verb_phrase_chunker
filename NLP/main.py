import nltk
from nltk.corpus import conll2000
from nltk.chunk.regexp import *
from nltk.chunk.util import *
from ChunkerClass import ChunkerClass

chunked_sents = conll2000.chunked_sents('test.txt', chunk_types=['VP']) + conll2000.chunked_sents('train.txt', chunk_types=['VP'])
tagged_sents = conll2000.tagged_sents('test.txt') + conll2000.tagged_sents('train.txt')

chunker = ChunkerClass()
chunker.chunk(chunked_sents, tagged_sents)
chunker.evaluate()

