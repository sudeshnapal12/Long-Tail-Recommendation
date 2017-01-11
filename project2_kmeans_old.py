import sys
from collections import Counter
from pprint import pprint

import itertools

try:
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from pyspark.sql import SQLContext, SparkSession
    from pyspark.mllib.feature import HashingTF
    from pyspark.mllib.feature import IDF
    from pyspark.mllib.linalg import SparseVector, DenseVector
    from pyspark.mllib.clustering import KMeans, KMeansModel

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

metadata = "meta_Video_Games_small.json"
ratingsFile = "ratings_Video_Games.10k.csv"
sc = SparkContext()
ss = SparkSession.builder.getOrCreate()

def getAvgRating(ratingList):
    sum = 0
    for rating in ratingList:
        sum += float(rating)
    avg = sum / len(ratingList)
    return avg

def getHeadTail(x, longtail_thrs, max_ratings):
    if(len(x[1]) > longtail_thrs*max_ratings):
        return 'HEAD'
    else:
        return 'TAIL'

longtail_thrs = 0.05
ratings = sc.textFile(ratingsFile).map(lambda line: (line.split(','))).cache()
item_user = ratings.map(lambda x: (x[1], x[0])).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)
item_popularity = item_user.map(lambda x: (x[0], len(x[1]))).sortBy(lambda x: -x[1]).collect()
max_ratings = item_popularity[0][1]

ratingsRdd = sc.textFile(ratingsFile) \
    .map(lambda x: x.split(",")) \
    .map(lambda x: (x[1], [x[2]])) \
    .reduceByKey(lambda x, y: x + y) \
    .map(lambda x: (x[0], [getHeadTail(x, longtail_thrs, max_ratings), getAvgRating(x[1]), len(x[1])]))
    # .sortBy(lambda x: -x[1][2])

pprint(ratingsRdd.collect())

metadataDF = ss.read.json(metadata)
# people.printSchema()

metadataDF.createOrReplaceTempView("metadata_table")
products = ss.sql("Select asin, categories from metadata_table ").na.drop()
# products.show(n=20)
productsRdd = products.rdd.map(lambda x: tuple(x))
# pprint(productsRdd.count())

itemRDD = ratingsRdd.join(productsRdd)
pprint(itemRDD.collect())

categoriesRDD = itemRDD.map(lambda x: (x[1][1])).map(lambda x : str(x))
pprint(categoriesRDD.collect())
numClusters = 20
hashingTF = HashingTF()
tf = hashingTF.transform(categoriesRDD)
# tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)

idfIgnore = IDF(minDocFreq=2).fit(tf)
tfidfIgnore = idfIgnore.transform(tf)
pprint(tfidfIgnore)

parsedData = tfidfIgnore.map(lambda x: DenseVector(x.toArray()))
# pprint(parsedData.collect())
model = KMeans.train(parsedData, 5)
print("Final centers: " + str(model.clusterCenters))
print("Total Cost: " + str(model.computeCost(parsedData)))
print(model.predict(parsedData).collect())
