import sys
from collections import Counter
from pprint import pprint
from nltk.corpus import stopwords

import itertools

try:
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    from pyspark.sql import SQLContext, SparkSession
    from pyspark.mllib.feature import HashingTF
    from pyspark.mllib.feature import IDF
    from pyspark.mllib.linalg import SparseVector, DenseVector
    from pyspark.mllib.clustering import KMeans, KMeansModel
    import numpy as np

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

metadata = "meta_Video_Games.json"
ratingsFile = "ratings_Video_Games.csv"
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

def checkIfHead(item):
    if item[0][1][0] == "HEAD":
        pprint("head")
        return (item[0][0],item[1])
    return None


def cosine(v0, v1):
    dot_product = 0.0
    card_v0 = 0.0
    card_v1 = 0.0
    for i in range(0, len(v1)):
        dot_product += float(v0[i]) * float(v1[i])
        card_v0 += float(v0[i]) ** 2
        card_v1 += float(v1[i]) ** 2
    if (card_v0 * card_v1) > 0:
        return dot_product / (card_v0 * card_v1)
    else:
        return 0


def calSimThres(x):
    avg = sum(x) / len(x)
    std = np.std(x)
    return (avg + 0.3 * std)


longtail_thrs = 0.5
numClusters=5

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

metadataDF = ss.read.json(metadata)
# people.printSchema()

metadataDF.createOrReplaceTempView("metadata_table")
products = ss.sql("Select asin, categories from metadata_table ").na.drop()
# products.show(n=20)
productsRdd = products.rdd.map(lambda x: tuple(x))
# pprint(productsRdd.count())

itemRDD = ratingsRdd.join(productsRdd)
# pprint(itemRDD.collect())

categoriesRDD = itemRDD.map(lambda x: (x[0], x[1][1])).map(lambda x : (x[0], str(x[1])))
categoriesOnlyRDD = categoriesRDD.map(lambda x: x[1])

hashingTF = HashingTF()
tf = hashingTF.transform(categoriesOnlyRDD)
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
idfIgnore = IDF(minDocFreq=2).fit(tf)
tfidfIgnore = idfIgnore.transform(tf)

# tf = categoriesRDD.mapValues(lambda x: hashingTF.transform(x))
# tf.cache()
# idf = IDF().fit(tf.values())
# tfidf = idf.transform(tf)

parsedData = tfidfIgnore.map(lambda x: DenseVector(x.toArray()))
model = KMeans.train(parsedData, numClusters)
# pprint("Final centers: " + str(model.clusterCenters))
# pprint("Total Cost: " + str(model.computeCost(parsedData)))
predictions = model.predict(parsedData)
# FORMAT returned => (item_id, ['HEAD', avg_rating, #count], cluster_id)
labelsAndPredictions = itemRDD.map(lambda x: [x[0], x[1][0]]).zip(predictions)

desc = ss.sql("Select asin, categories, description from metadata_table ").na.drop()
descRdd = desc.rdd.map(lambda x: tuple(x))
itemDescRDD = ratingsRdd.join(descRdd)
# pprint(itemDescRDD.collect())
descRDD = itemDescRDD.map(lambda x: (x[0], x[1][1])).map(lambda x : (x[0], str(x[1])))

cachedStopWords = stopwords.words("english")
def removeStopWord(desc):
    newDesc = ' '.join([word for word in desc.split() if word not in cachedStopWords])
    return newDesc

descOnlyRDD = descRDD.map(lambda x: x[1])\
    .map(lambda x: removeStopWord(x))
# pprint(descOnlyRDD.collect())

hashingTF = HashingTF()
tf = hashingTF.transform(descOnlyRDD)
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
idfIgnore = IDF(minDocFreq=2).fit(tf)
tfidfIgnore = idfIgnore.transform(tf)
descTFIDFVec = descRDD.map(lambda x: x[0]).zip(tfidfIgnore)


pprint("test")
# pprint(labelsAndPredictions.collect())#TODO uncomment

headitemsRDD = labelsAndPredictions.map(lambda x:checkIfHead(x))\
            .filter(lambda x:x!=None)


headItemsList = headitemsRDD.collect()
pprint(len(headItemsList))

tailCount = 0
totalCount = 0

for item in headItemsList:
    targetHeadItemID = item[0] 
    targetHeadItem = labelsAndPredictions.filter(lambda x: x[0][0] == targetHeadItemID)
    targetHeadItemCID = targetHeadItem.map(lambda x: x[1]).collect()[0]
    print('TARGET CLUSTER ID', targetHeadItemCID)

    itemsInClusterWithTarget = labelsAndPredictions.filter(lambda x: x[1] == targetHeadItemCID).map(lambda x: (x[0][0], x[0][1]))
    pprint("cluster Items")
    pprint(itemsInClusterWithTarget.collect())
    ## Recommend items with similar description

    descTFIDFVecTrim = itemsInClusterWithTarget.join(descTFIDFVec)

    descTFIDFVecOnly = descTFIDFVecTrim \
        .map(lambda x: (x[0], [x[1][0][0], x[1][1]])) \
        .map(lambda x: (x[0], [x[1][0], DenseVector(x[1][1].toArray())]))
    targetVec = descTFIDFVecOnly \
        .filter(lambda x: x[0] == targetHeadItemID) \
        .map(lambda x: x[1][1]).collect()

    neighbourVec = descTFIDFVecOnly \
        .filter(lambda x: x[0] != targetHeadItemID)
    # pprint(neighbourVec.collect())

    # pprint(otherVec.collect())


    itemTargetSim = neighbourVec \
        .map(lambda x: (x[0], [x[1][0], cosine(x[1][1], targetVec[0])]))

    simThresList = itemTargetSim \
        .map(lambda x: x[1][1]).collect()
    simThres = calSimThres(simThresList)

    ## Recommended head or tail items with similarity greater than similarity threshold
    ## return (itemid, 'TAIL')
    itemRec = itemTargetSim \
        .filter(lambda x: x[1][1] > simThres) \
        .map(lambda x: (x[0], x[1][0]))


    recommendedList = itemRec.collect()
    pprint(recommendedList)
    totalCount = len(recommendedList)
    for item in recommendedList:
        if item[1] == "TAIL":
            tailCount +=1
    pprint(tailCount / totalCount)
    tailCount=0

sc.stop()