from gensim import corpora, similarities

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
# 遍历documents，将其每个元素的words置为小写，然后通过空格分词，并过滤掉在stoplist中的word。
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
# remove words that appear only once，collection是python的一个工具库
from collections import defaultdict

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

'''from pprint import pprint  # pprint可以使输出更易观看。
pprint(texts)'''

# 定义一个词典，里面包含所有语料库中的单词，这里假设上文中输出的texts就是经过处理后的语料库。
dictionary = corpora.Dictionary(texts)
dictionary.save('./out/deerwester.dict')  # 因为实际运用中该词典非常大，所以将训练的词典保存起来，方便将来使用。
# print(dictionary)
# dictionary有35个不重复的词，给每个词赋予一个id
# print(dictionary.token2id)

new_doc = "Human computer interaction"
# 用dictionary的doc2bow方法将文本向量化
new_vec = dictionary.doc2bow(new_doc.lower().split())
# corpora.MmCorpus.serialize('./out/deerwester.mm',new_vec)  # 讲训练结果存储到硬盘中，方便将来使用。
# print(new_vec)

corpus = [dictionary.doc2bow(text) for text in texts]


# corpora.MmCorpus.serialize('/out/deerwester.mm', corpus) # 存入硬盘，以备后需
# print(corpus)

# 获取语料
class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # 每一个line代表语料库中的一个文档
            yield dictionary.doc2bow(line.lower().split())


corpus_memory_friendly = MyCorpus()  # 没有将corpus加载到内存中
# print(corpus_memory_friendly)#输出：<__main__.MyCorpus object at 0x10d5690>

# 遍历每个文档
# for vector in corpus_memory_friendly:  # load one vector into memory at a time
#     print(vector)


from gensim import models

tfidf = models.TfidfModel(corpus)

doc_bow = [(0, 1), (1, 1)]
# print(tfidf[doc_bow])


# 构造LSI模型并将待检索的query和文本转化为LSI主题向量
# 转换之前的corpus和query均是BOW向量
lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
documents = lsi_model[corpus]
query_vec = lsi_model[corpus]

index = similarities.MatrixSimilarity(documents)
sims = index[query_vec]  # return: an iterator of tuple (idx, sim)

print(sims)
