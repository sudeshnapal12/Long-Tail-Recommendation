from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from numpy import array, corrcoef
from pprint import pprint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import SparseVector, DenseVector

from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

# sc = SparkContext()
ratingsFile = sc.textFile('sample_metadata.json')
ratings = ratingsFile.map(lambda line: (line.split(':')))
pprint(type(ratingsFile))

# item_user = ratings.map(lambda x: (x[-1], x[0])).map(lambda x: (x[0], [x[-1]])).reduceByKey(lambda x, y: x + y).collect()
item_user = ratings.map(lambda x: x[-1])

# pprint(item_user)

# item_user = item_user[1]

numClusters = 20
hashingTF = HashingTF()
tf = hashingTF.transform(item_user)
# tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

idfIgnore = IDF(minDocFreq=2).fit(tf)
tfidfIgnore = idfIgnore.transform(tf)
#pprint(tfidfIgnore)
tfidfIgnoreDense = tfidfIgnore.map(lambda x: DenseVector(x.toArray()))
corpus = tfidfIgnoreDense.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
ldaModel = LDA.train(corpus, k=3)
print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize())
      + " words):")

topics = ldaModel.topicsMatrix()
# for topic in range(3):
#     print("Topic " + str(topic) + ":")
#     for word in range(0, ldaModel.vocabSize()):
#         print(" " + str(topics[word][topic]))

# pprint(corpus.take(1))

parsedData = tfidfIgnore.map(lambda x: x.toArray())
pprint(parsedData.take(1))
# gmm = GaussianMixture.train(parsedData, 2)
# for i in range(2):
#     print("weight = ", gmm.weights[i], "mu = ", gmm.gaussians[i].mu,
#           "sigma = ", gmm.gaussians[i].sigma.toArray())


# parsedData = tfidfIgnore.map(lambda p: (p.split(',')))
# corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

# pprint(parsedData)


# def matrixToRDD(m: Matrix): RDD[Vector] = {
# val columns = m.toArray.grouped(m.numRows)
# val rows = columns.toSeq.transpose val vectors = rows.map(row => new DenseVector(row.toArray))
# sc.parallelize(vectors)
# }
# sc = SparkContext()

#ratingsFile = sc.textFile('sample_metadata.json')
#ratings = ratingsFile.map(lambda line: (line.split(','))).cache()
# ratings1 = Array(ratings)
# Vectors.dense(ratings)
#corrs = ratings.cartesian(ratings).map(lambda (x,y): corrcoef(x,y)[0,1]).collect()
#sc.stop()