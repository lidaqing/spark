spark-shell --master spark://master:7077 --driver-memory 1G --total-executor-cores 2

import org.apache.log4j.{Level,Logger}
import org.apache.spark.mllib.recommendation.{ALS,MatrixFactorizationModel,Rating}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.rdd._

Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
Logger.getLogger("org.apache.eclipse.jetty.server").setLevel(Level.OFF)

val rawratings = sc.textFile("hdfs://master:9000/home/hadoop/data/u.data").map(_.split("\t").take(3))

val ratings = rawratings.map{case Array(user,movie,rating) => Rating(user.toInt,movie.toInt,rating.toDouble)}.cache()

val movies= sc.textFile("hdfs://master:9000/home/hadoop/data/u.item").map(_.split("\\|").take(2))

val numRatings = ratings.count()
val numUsers = ratings.map(_.user).distinct().count()
val numMovies = ratings.map(_.product).distinct().count()

println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")
val splits = ratings.randomSplit(Array(0.6,0.2,0.2),12)
val training = splits(0).cache()
val validation = splits(1).cache()
val test = splits(1).cache()
val numTraining = training.count()
val numValidation = validation.count()
val numTest = test.count()

println("Training: " + numTraining + validation:" + numValidation + " test: " + numTest)

def computeRmse(model:MatrixFactorizationModel,data:RDD[Rating],n:Long):Double=
{
    val predictions:RDD[Rating] = model.predict((data.map(x => (x.user,x.product))))
    val predictionsAndRatings = predictions.map
    {
        x => ((x.user,x.product),x.rating)
    }.join(data.map(x => ((x.user,x.product),x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2)*(x._1 - x._2)).reduce(_+_)/n)
}

val ranks = List(20,40)
val lambdas = List(0.1,10.0)
val numIters = List(5,10)
var bestModel:Option[MatrixFactorizationModel] = None
var bestValidationRmse = Double.MaxValue
var bestRank = 0
var bestLambda = 1.0
var bestNumIter = 1

for(rank <- ranks; lambda <- lambdas; numIter <- numIters)
{
    val model = ALS.train(training,rank,numIter,lambda)
    val validationRmse = computeRmse(model,validation,numValidation)
    println("RMSE(validation)=" +validationRmse+ "for the model trained with rank = " + rank + ",lambda = " +lambda + ",and numIter = "+numIter + ".")
    if(validationRmse < bestValidationRmse)
    {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
    }
}

val testRmse = computeRmse(bestModel.get,test,numTest)
println("The best model was trained with rank = "+bestRank + " and lambda =" + bestLambda
+ ",and numIter =" +bestNumIter +", and its RMSE on the test set is " + testRmse +".")

val meanRating = training.union(validation).map(_.rating).mean
val baselineRmse = math.sqrt(test.map(x => (meanRating - x.rating)*(meanRating - x.rating)).reduce(_+_)/numTest)
val improvement = (baselineRmse - testRmse)/baselineRmse*100
println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%. ")

val userId = 789
val K =10
val topKRecs = bestModel.get.recommendProducts(userId,K)
topKRecs.foreach(println)

val titles = movies.map(array => (array(0).toInt,array(1))).collectAsMap()
topKRecs.map(rating => (titles(rating.product),rating.rating)).foreach(println)

val moviesForUser = ratings.keyBy(_.user).lookup(789)
val titles = movies.map(array => (array(0).toInt,array(1))).collectAsMap()
println(moviesForUser.size)
moviesForUser.sortBy(_.rating).take(10).map(rating => (titles(rating.product),rating.rating)).foreach(println)
