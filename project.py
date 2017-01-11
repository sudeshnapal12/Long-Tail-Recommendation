from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
from itertools import combinations
import itertools
import numpy as np
from pprint import pprint
from itertools import chain
from operator import add
from math import sqrt
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel

import matplotlib.pyplot as plt

sc = SparkContext()
ratingsFile = sc.textFile('ratings_Video_Games.10k.csv')

# returns (user, item, rating, time)
ratings = ratingsFile.map(lambda line: (line.split(','))).cache()

#check ratings distribution
# ratings_dist = ratings.map(lambda x: (x[2], x[1])).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)\
#     .map(lambda x: (x[0], len(x[1]))).sortBy(lambda x: x[0]).collect()
# pprint(ratings_dist)
# item = zip(*ratings_dist)[0]
# count = zip(*ratings_dist)[1]
# x_pos = np.arange(len(item))
# slope, intercept = np.polyfit(x_pos, count, 1)
# plt.bar(x_pos, count,align='center')
# plt.xticks(x_pos, item)
# plt.ylabel('Number of reviews')
# plt.xlabel('Rating')
# plt.show()

item_user = ratings.map(lambda x: (x[1], x[0])).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)
item_popularity = item_user.map(lambda x: (x[0], len(x[1]))).sortBy(lambda x: -x[1]).collect()
max_ratings = item_popularity[0][1]
print max_ratings

def findItemFeatures(x, longtail_thrs, max_ratings):
    rating_sum = 0.0
    for rating in x[1]:
        rating_sum += float(rating)
    rating_mean = float(rating_sum) / len(x[1])
    if(len(x[1]) > longtail_thrs*max_ratings):
        return (x[0], (rating_mean, float(len(x[1])), 'HEAD'))
    else:
        return (x[0], (rating_mean, float(len(x[1])), 'TAIL'))

longtail_thrs = 0.05
item_ftrs_map = ratings.map(lambda x:(x[1], x[2])).map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y)\
    .map(lambda x: findItemFeatures(x, longtail_thrs, max_ratings)).sortBy(lambda x: -x[1][1])

pprint(item_ftrs_map.mapValues(list).collect())
head_ratings_dist = item_ftrs_map.filter(lambda x:x[1][2] == 'TAIL').map(lambda x: (x[1][0], x[1][1]))
pprint(head_ratings_dist.collect())


pprint(item_ftrs_map.collect())

# Clustering
# item_ftrs = item_ftrs_map.map(lambda x: np.array([x[1][0], x[1][1]]))
# # item_ftrs = item_ftrs_map.map(lambda x: np.array(x[1]))
# model = KMeans.train(item_ftrs, 10)
# print("Final centers: " + str(model.clusterCenters))
# print("Total Cost: " + str(model.computeCost(item_ftrs)))

# pprint(item_mean_rating.collect())
#
#
# # # # plot longtail with matplotlib after bringing RDD to local
# # # item = zip(*item_popularity)[0]
# # # count = zip(*item_popularity)[1]
# # # x_pos = np.arange(len(item))
# # # slope, intercept = np.polyfit(x_pos, count, 1)
# # # plt.bar(x_pos, count,align='center')
# # # plt.xticks(x_pos, item)
# # # plt.ylabel('Popularity')
# # # plt.xlabel('Items')
# # # plt.show()