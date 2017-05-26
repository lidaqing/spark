import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

case class Rating(userId:Int,movieId:Int,rating:Float,timestamp:Long)

def parseRating(str:String):Rating = {
    val fields = str.split("\t")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt,fields(2).toFloat,fields(3).toLong)
}

val ratings = spark.read.textFile("hdfs://master:9000/home/hadoop/data/u.data").map(parseRating).toDF().cache()
ratings.describe("userId","movieId","rating").show()

+-------+------------------+------------------+------------------+
|summary|            userId|           movieId|            rating|
+-------+------------------+------------------+------------------+
|  count|            100000|            100000|            100000|
|   mean|         462.48475|         425.53013|           3.52986|
| stddev|266.61442012750905|330.79835632558473|1.1256735991443214|
|    min|                 1|                 1|               1.0|
|    max|               943|              1682|               5.0|
+-------+------------------+------------------+------------------+

val Array(training,test) = ratings.randomSplit(Array(0.8,0.2))
val als = new ALS().setMaxIter(5).setRank(10).setRegParam(0.01).setImplicitPrefs(false).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = als.fit(training)
val predictions = model.transform(test)
val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")
val prediction11 = predictions.na.drop()
predictions.filter(predictions("prediction").isNaN).select("userId","movieId","rating","prediction").limit(10).show()

+------+-------+------+----------+                                              
|userId|movieId|rating|prediction|
+------+-------+------+----------+
|   405|   1580|   1.0|       NaN|
|   655|   1650|   4.0|       NaN|
|   381|   1533|   4.0|       NaN|
|   279|   1500|   5.0|       NaN|
|   781|   1500|   5.0|       NaN|
|   303|   1510|   3.0|       NaN|
|    76|   1156|   3.0|       NaN|
|   655|   1633|   3.0|       NaN|
|     7|    599|   1.0|       NaN|
|   291|   1505|   4.0|       NaN|
+------+-------+------+----------+

val rmse = evaluator.evaluate(prediction11)

#rmse: Double = 1.0726353698264208 

println(s"Root-mean-square error = $rmse")

#Root-mean-square error = 1.0726353698264208

prediction11.createOrReplaceTempView("tmp_predictions")
val absDiff = spark.sql("select abs(prediction-rating) as diff from tmp_predictions")

#absDiff: org.apache.spark.sql.DataFrame = [diff: float]

absDiff.createOrReplaceTempView("tmp_absDiff")
spark.sql("select mean(diff) as absMeanDiff from tmp_absDiff").show()

+------------------+                                                            
|       absMeanDiff|
+------------------+
|0.8135789327773422|
+------------------+
