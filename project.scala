import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD


/**
  * Created by dexter on 12/16/2016.
  */
object project {

  def getHeadTail( a:Int, longtail_thrs:Double, max_ratings:Int ): String = {
    if(a > longtail_thrs*max_ratings) {
      "HEAD"
    }else{
      "TAIL"
    }
  }

  def main(args: Array[String]) {
    val ratingsFile = "ratings_Video_Games.10k.csv"
    val metadataFile = "meta_Video_Games.json"
    val conf = new SparkConf().setMaster("local[2]").setAppName("Hello Spark")
    val sc = new SparkContext(conf)
    val ss = SparkSession.builder.getOrCreate()

    val ratings = sc.textFile(ratingsFile).map(line => (line.split(','))).cache()
    val item_user = ratings.map(x => (x(1), x(0))).groupByKey().mapValues(_.toList)
    val item_popularity = item_user.map(x => (x._1, x._2.size)).sortBy(x => -x._2).collect()

    val max_ratings = item_popularity(0)._2
    val longtail_thrs = 0.05
    val ratingsRdd = ratings.map(x => (x(1), x(2))).groupByKey().map(x => (x._1, List(getHeadTail(x._2.size, longtail_thrs, max_ratings), x._2.size)))

    val metadataDF = ss.read.json(metadataFile)

    metadataDF.createOrReplaceTempView("metadata_table")
    val items = ss.sql("Select asin, categories from metadata_table ").na.drop()
    val itemsRdd = items.rdd.map(x => (x(0).toString, List(x(1))))
    val itemNewRDD = (ratingsRdd union itemsRdd).reduceByKey(_ ++ _)
    val categories = itemNewRDD.map(x => x._2)

    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(categories)
    tf.cache()
    val idf = new IDF().fit(tf)
    val tfidf: RDD[Vector] = idf.transform(tf)
    val idfIgnore = new IDF(minDocFreq = 2).fit(tf)
    val tfidfIgnore: RDD[Vector] = idfIgnore.transform(tf)

    itemsRdd.collect().foreach(println)
    ratingsRdd.collect().foreach(println)
    itemNewRDD.collect().foreach(println)
    categories.collect().foreach(println)


    println("END")
  }

}
