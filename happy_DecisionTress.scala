//spark-shell --master spark://master:7077 --driver-memory 1G --total-executor-cores 2

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString,StringIndexer,VectorIndexer,VectorAssembler}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticalssClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder,CrossValidator}
import spark.implicits._

//导入数据
val data = spark.read.format("csv").option("header",true).load("hdfs://master:9000/home/hadoop/data/happy01.csv")
data.show(5)

+-------+------+---+------------+--------+-------------+---------+----------+------+
|affairs|gender|age|yearsmarried|children|religiousness|education|occupation|rating|
+-------+------+---+------------+--------+-------------+---------+----------+------+
|      0|  male| 37|          10|      no|            3|     null|         7|     4|
|      0|female| 27|           4|      no|            4|     null|         6|     4|
|      0|female| 32|          15|     yes|            1|     null|         1|     4|
|      0|  male| 57|          15|     yes|            5|       18|         6|     5|
|      0|  male| 22|        0.75|      no|            2|       17|         6|     3|
+-------+------+---+------------+--------+-------------+---------+----------+------+

data.printSchema()

root
 |-- affairs: string (nullable = true)
 |-- gender: string (nullable = true)
 |-- age: string (nullable = true)
 |-- yearsmarried: string (nullable = true)
 |-- children: string (nullable = true)
 |-- religiousness: string (nullable = true)
 |-- education: string (nullable = true)
 |-- occupation: string (nullable = true)
 |-- rating: string (nullable = true)


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
data1.describe("affairs","age","yearsmarried","rating").show()

+-------+------------------+-----------------+-----------------+------------------+
|summary|           affairs|              age|     yearsmarried|            rating|
+-------+------------------+-----------------+-----------------+------------------+
|  count|               601|              601|              601|               601|
|   mean|1.4559068219633944|32.48752079866888| 8.17769550748752|3.9317803660565724|
| stddev|3.2987577284946816| 9.28876170487667|5.571303149963791|1.1031794920503797|
|    min|               0.0|             17.5|            0.125|               1.0|
|    max|              12.0|             57.0|             15.0|               5.0|
+-------+------------------+-----------------+-----------------+------------------+


val label_ch = "case when affairs = 0 then 0 else cast(1 as double) end as label"
val gender_ch = "case when gender = 'female' then 0 else cast(1 as double) end as gender"
val children_ch = "case when children = 'no' then 0 else cast(1 as double) end as children"
val education_ch = "ifnull(education,14) as education"
val dataLabelDF = spark.sql(s"select $label_ch,$gender_ch,age,yearsmarried,$children_ch,religiousness,$education_ch,occupation,rating from data")
dataLabelDF.show(5)

+-----+------+----+------------+--------+-------------+---------+----------+------+
|label|gender| age|yearsmarried|children|religiousness|education|occupation|rating|
+-----+------+----+------------+--------+-------------+---------+----------+------+
|  0.0|   1.0|37.0|        10.0|     0.0|          3.0|     14.0|       7.0|   4.0|
|  0.0|   0.0|27.0|         4.0|     0.0|          4.0|     14.0|       6.0|   4.0|
|  0.0|   0.0|32.0|        15.0|     1.0|          1.0|     14.0|       1.0|   4.0|
|  0.0|   1.0|57.0|        15.0|     1.0|          5.0|     18.0|       6.0|   5.0|
|  0.0|   1.0|22.0|        0.75|     0.0|          2.0|     17.0|       6.0|   3.0|
+-----+------+----+------------+--------+-------------+---------+----------+------+


//生成存放特征值的特征向量
val featuresArray = Array("gender","age","yearsmarried","children","religiousness","education","occupation","rating")
//把源数据组合成特征向量features
val assembler = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")
val vecDF = assembler.transform(dataLabelDF)
vecDF.show(5,false)

+-----+------+----+------------+--------+-------------+---------+----------+------+------------------------------------+
|label|gender|age |yearsmarried|children|religiousness|education|occupation|rating|features                            |
+-----+------+----+------------+--------+-------------+---------+----------+------+------------------------------------+
|0.0  |1.0   |37.0|10.0        |0.0     |3.0          |14.0     |7.0       |4.0   |[1.0,37.0,10.0,0.0,3.0,14.0,7.0,4.0]|
|0.0  |0.0   |27.0|4.0         |0.0     |4.0          |14.0     |6.0       |4.0   |[0.0,27.0,4.0,0.0,4.0,14.0,6.0,4.0] |
|0.0  |0.0   |32.0|15.0        |1.0     |1.0          |14.0     |1.0       |4.0   |[0.0,32.0,15.0,1.0,1.0,14.0,1.0,4.0]|
|0.0  |1.0   |57.0|15.0        |1.0     |5.0          |18.0     |6.0       |5.0   |[1.0,57.0,15.0,1.0,5.0,18.0,6.0,5.0]|
|0.0  |1.0   |22.0|0.75        |0.0     |2.0          |17.0     |6.0       |3.0   |[1.0,22.0,0.75,0.0,2.0,17.0,6.0,3.0]|
+-----+------+----+------------+--------+-------------+---------+----------+------+------------------------------------+


//对label进行索引或重新编码
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").setHandleInvalid("skip").fit(vecDF)
//自动识别分类的特征，并特征值进行索引，对具有大于8个不同的值的特征被视为连续
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(8).fit(vecDF)
//查看分类信息
featureIndexer.transform(vecDF).show(5,false)

+-----+------+----+------------+--------+-------------+---------+----------+------+------------------------------------+----------------------------------+
|label|gender|age |yearsmarried|children|religiousness|education|occupation|rating|features                            |indexedFeatures                  |
+-----+------+----+------------+--------+-------------+---------+----------+------+------------------------------------+----------------------------------+
|0.0  |1.0   |37.0|10.0        |0.0     |3.0          |14.0     |7.0       |4.0   |[1.0,37.0,10.0,0.0,3.0,14.0,7.0,4.0]|[1.0,37.0,6.0,0.0,2.0,2.0,6.0,3.0]|
|0.0  |0.0   |27.0|4.0         |0.0     |4.0          |14.0     |6.0       |4.0   |[0.0,27.0,4.0,0.0,4.0,14.0,6.0,4.0] |[0.0,27.0,4.0,0.0,3.0,2.0,5.0,3.0]|
|0.0  |0.0   |32.0|15.0        |1.0     |1.0          |14.0     |1.0       |4.0   |[0.0,32.0,15.0,1.0,1.0,14.0,1.0,4.0]|[0.0,32.0,7.0,1.0,0.0,2.0,0.0,3.0]|
|0.0  |1.0   |57.0|15.0        |1.0     |5.0          |18.0     |6.0       |5.0   |[1.0,57.0,15.0,1.0,5.0,18.0,6.0,5.0]|[1.0,57.0,7.0,1.0,4.0,5.0,5.0,4.0]|
|0.0  |1.0   |22.0|0.75        |0.0     |2.0          |17.0     |6.0       |3.0   |[1.0,22.0,0.75,0.0,2.0,17.0,6.0,3.0]|[1.0,22.0,2.0,0.0,1.0,4.0,5.0,2.0]|
+-----+------+----+------------+--------+-------------+---------+----------+------+------------------------------------+----------------------------------+

//训练模型
//将数据分为训练和测试集
val Array(trainingData,testData) = vecDF.randomSplit(Array(0.8,0.2),12)
//训练决策树模型
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setImpurity("entropy")
.setMaxBins(100).setMaxDepth(5).setMinInfoGain(0.01).setMinInstancesPerNode(10).setSeed(123)
//将索引标签转换回原始标签，其中prediction为自动产生的预测label行名字
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//把所有特征及模型放在一个管道中
val pipeline = new Pipeline().setStages(Array(labelIndexer,featureIndexer,dt,labelConverter))

//训练模型
val model = pipeline.fit(trainingData)
//作出预测
val predictions = model.transform(testData)
//选择几个实例行展示
predictions.select("predictedLabel","label","features").show(10,truncate=false)

+--------------+-----+-------------------------------------+
|predictedLabel|label|features                             |
+--------------+-----+-------------------------------------+
|0.0           |0.0  |[0.0,22.0,0.125,0.0,4.0,12.0,4.0,5.0]|
|0.0           |0.0  |[0.0,22.0,0.417,0.0,4.0,14.0,5.0,5.0]|
|0.0           |0.0  |[0.0,22.0,0.417,0.0,5.0,14.0,1.0,4.0]|
|0.0           |0.0  |[0.0,22.0,0.75,0.0,2.0,16.0,3.0,4.0] |
|0.0           |0.0  |[0.0,22.0,0.75,0.0,5.0,18.0,1.0,5.0] |
|0.0           |0.0  |[0.0,22.0,1.5,0.0,1.0,16.0,6.0,5.0]  |
|0.0           |0.0  |[0.0,22.0,1.5,0.0,2.0,16.0,4.0,5.0]  |
|0.0           |0.0  |[0.0,22.0,1.5,0.0,2.0,17.0,5.0,4.0]  |
|0.0           |0.0  |[0.0,22.0,1.5,0.0,2.0,18.0,5.0,5.0]  |
|0.0           |0.0  |[0.0,22.0,1.5,0.0,3.0,16.0,6.0,5.0]  |
+--------------+-----+-------------------------------------+


//评估模型
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)

accuracy: Double = 0.782608695652174

print("Test Error = " + (1.0 -accuracy))

Test Error = 0.21739130434782605

