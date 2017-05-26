//spark-shell --master spark://master:7077 --driver-memory 1G --total-executor-cores 2

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary,LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder,IndexToString,StringIndexer,VectorIndexer,VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder,CrossValidator}
import org.apache.spark.ml.feature.StandardScaler
import spark.implicits._

//导入数据
val data = spark.read.format("csv").option("header",true).load("hdfs://master:9000/home/hadoop/data/happy01.csv")
data.show(5)
data.printSchema()

//数据预处理，转换字符类型
val data1 = data.select(
    data("affairs").cast("Double"),
    data("age").cast("Double"),
    data("yearsmarried").cast("Double"),
    data("religiousness").cast("Double"),
    data("education").cast("Double"),
    data("occupation").cast("Double"),
    data("rating").cast("Double"),
    data("gender").cast("String"),
    data("children").cast("String")
)
data1.printSchema()
data1.createOrReplaceTempView("data")
data.describe("affairs","age","yearsmarried","rating").show()

+-------+------------------+-----------------+-----------------+------------------+
|summary|           affairs|              age|     yearsmarried|            rating|
+-------+------------------+-----------------+-----------------+------------------+
|  count|               601|              601|              601|               601|
|   mean|1.4559068219633944|32.48752079866888| 8.17769550748752|3.9317803660565724|
| stddev|3.2987577284946816| 9.28876170487667|5.571303149963791|1.1031794920503797|
|    min|                 0|             17.5|            0.125|                 1|
|    max|                 7|               57|                7|                 5|
+-------+------------------+-----------------+-----------------+------------------+


val label_ch = "case when affairs=0 then 0 else cast(1 as double) end as label"
val education_ch = "ifnull(education,14) as education"
val dataLabelDF = spark.sql(s"select $label_ch,gender,age,yearsmarried,children,religiousness,$education_ch,occupation,rating from data")

val indexedgender = new StringIndexer().setInputCol("gender").setOutputCol("indexGender").fit(dataLabelDF).transform(dataLabelDF)
val coderDf = new OneHotEncoder().setInputCol("indexGender").setOutputCol("gendervector").setDropLast(false).transform(indexedgender)

val indexedchildren = new StringIndexer().setInputCol("children").setOutputCol("indexChildren").fit(coderDf).transform(coderDf)
val coderDf1 = new OneHotEncoder().setInputCol("indexChildren").setOutputCol("childrenvector").setDropLast(false).transform(indexedchildren)

val encodeDF:DataFrame = coderDf1
encodeDF.show()

+-----+------+----+------------+--------+-------------+---------+----------+------+-----------+-------------+-------------+--------------+
|label|gender| age|yearsmarried|children|religiousness|education|occupation|rating|indexGender| gendervector|indexChildren|childrenvector|
+-----+------+----+------------+--------+-------------+---------+----------+------+-----------+-------------+-------------+--------------+
|  0.0|  male|37.0|        10.0|      no|          3.0|     14.0|       7.0|   4.0|        1.0|(2,[1],[1.0])|          1.0| (2,[1],[1.0])|
|  0.0|female|27.0|         4.0|      no|          4.0|     14.0|       6.0|   4.0|        0.0|(2,[0],[1.0])|          1.0| (2,[1],[1.0])|
|  0.0|female|32.0|        15.0|     yes|          1.0|     14.0|       1.0|   4.0|        0.0|(2,[0],[1.0])|          0.0| (2,[0],[1.0])|
|  0.0|  male|57.0|        15.0|     yes|          5.0|     18.0|       6.0|   5.0|        1.0|(2,[1],[1.0])|          0.0| (2,[0],[1.0])|
|  0.0|  male|22.0|        0.75|      no|          2.0|     17.0|       6.0|   3.0|        1.0|(2,[1],[1.0])|          1.0| (2,[1],[1.0])|


//将字段组合成向量feature
val assembler = new VectorAssembler().setInputCols(Array("age","yearsmarried","religiousness","education","occupation","rating","gendervector","childrenvector")).setOutputCol("features")
val vecDF = assembler.transform(encodeDF)
vecDF.select("features").show(5,false)

+--------------------------------------------+
|features                                    |
+--------------------------------------------+
|[37.0,10.0,3.0,14.0,7.0,4.0,0.0,1.0,0.0,1.0]|
|[27.0,4.0,4.0,14.0,6.0,4.0,1.0,0.0,0.0,1.0] |
|[32.0,15.0,1.0,14.0,1.0,4.0,1.0,0.0,1.0,0.0]|
|[57.0,15.0,5.0,18.0,6.0,5.0,0.0,1.0,1.0,0.0]|
|[22.0,0.75,2.0,17.0,6.0,3.0,0.0,1.0,0.0,1.0]|
+--------------------------------------------+


//标准化-均值标准差
val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)
val scalerModel = scaler.fit(vecDF)

//正则化每个特征
val scaledData:DataFrame = scalerModel.transform(vecDF)
scaledData.select("features","scaledFeatures").show

+--------------------+--------------------+
|            features|      scaledFeatures|
+--------------------+--------------------+
|[37.0,10.0,3.0,14...|[0.48579986705462...|
|[27.0,4.0,4.0,14....|[-0.5907698973252...|
|[32.0,15.0,1.0,14...|[-0.0524850151353...|
|[57.0,15.0,5.0,18...|[2.63893939581439...|
|[22.0,0.75,2.0,17...|[-1.1290547795152...|
|[32.0,1.5,2.0,17....|[-0.0524850151353...|
|[22.0,0.75,2.0,12...|[-1.1290547795152...|
|[57.0,15.0,2.0,14...|[2.63893939581439...|
|[32.0,15.0,4.0,16...|[-0.0524850151353...|
|[22.0,1.5,4.0,14....|[-1.1290547795152...|
|[37.0,15.0,2.0,20...|[0.48579986705462...|
|[27.0,4.0,4.0,18....|[-0.5907698973252...|
|[47.0,15.0,5.0,17...|[1.56236963143450...|
|[22.0,1.5,2.0,17....|[-1.1290547795152...|
|[27.0,4.0,4.0,14....|[-0.5907698973252...|
|[37.0,15.0,1.0,17...|[0.48579986705462...|
|[37.0,15.0,2.0,18...|[0.48579986705462...|
|[22.0,0.75,3.0,16...|[-1.1290547795152...|
|[22.0,1.5,2.0,16....|[-1.1290547795152...|
|[27.0,10.0,2.0,14...|[-0.5907698973252...|
+--------------------+--------------------+


val Array(trainingData,testData) = scaledData.randomSplit(Array(0.8,0.2),seed=12345)

