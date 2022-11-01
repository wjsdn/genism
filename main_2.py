# iteritems用来遍历对象中的每个item
from gensim import corpora
from six import iteritems
stoplist = set('for a of the and to in'.split())
#初步构建所有单词的词典
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt') )
#去出停用词,stop_ids表示停用词在dictionary中的id
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
#只出现一次的单词id
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq ==1]
#根据stop_ids与once_ids清洗dictionary
dictionary.filter_tokens(stop_ids + once_ids)
# 去除清洗后的空位
dictionary.compactify()
print(dictionary)