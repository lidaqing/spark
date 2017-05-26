
//pyspark --master spark://master:7077 --driver-memory 1G --total-executor-cores 2

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer,VectorAssembler
from pyspark.ml.feature import OneHotEncoder,StringIndexer
from pyspark.ml.regression import DecisionTreeRegressionModel
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.sql.functions import *

#把数据导入spark
rawdata = spark.read.csv("hdfs://master:9000/home/hadoop/data/hour.csv",header=True)
rawdata.show(3)

+-------+----------+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+----------+---+
|instant|    dteday|season| yr|mnth| hr|holiday|weekday|workingday|weathersit|temp| atemp| hum|windspeed|casual|registered|cnt|
+-------+----------+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+----------+---+
|      1|2011-01-01|     1|  0|   1|  0|      0|      6|         0|         1|0.24|0.2879|0.81|        0|     3|        13| 16|
|      2|2011-01-01|     1|  0|   1|  1|      0|      6|         0|         1|0.22|0.2727| 0.8|        0|     8|        32| 40|
|      3|2011-01-01|     1|  0|   1|  2|      0|      6|         0|         1|0.22|0.2727| 0.8|        0|     5|        27| 32|
+-------+----------+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+----------+---+

rawdata.filter("season is null").select("season").show(3)


#选择特征
data1 = rawdata.select(
    rawdata.season.cast('double'),
    rawdata.yr.cast('double'),
    rawdata.mnth.cast('double'),
    rawdata.hr.cast('double'),
    rawdata.holiday.cast('double'),
    rawdata.weekday.cast('double'),
    rawdata.workingday.cast('double'),
    rawdata.weathersit.cast('double'),
    rawdata.temp.cast('double'),
    rawdata.atemp.cast('double'),
    rawdata.hum.cast('double'),
    rawdata.windspeed.cast('double'),
    rawdata.cnt.cast('double').alias("label")
)

#优化目标值
log_data = data1.select(data1.season,data1.yr,data1.mnth,data1.hr,data1.holiday,data1.weekday,data1.workingday,data1.weathersit,\
data1.temp,data1.atemp,data1.hum,data1.windspeed,round(log(data1.label),4).alias('label'))
log_data.show(2)

+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+
|season| yr|mnth| hr|holiday|weekday|workingday|weathersit|temp| atemp| hum|windspeed| label|
+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+
|   1.0|0.0| 1.0|0.0|    0.0|    6.0|       0.0|       1.0|0.24|0.2879|0.81|      0.0|2.7726|
|   1.0|0.0| 1.0|1.0|    0.0|    6.0|       0.0|       1.0|0.22|0.2727| 0.8|      0.0|3.6889|
+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+


#把前8个分类特征索引化
stringIndexer = StringIndexer(inputCol = "season",outputCol = "seasonindexed",handleInvalid = 'error').fit(data1)
td1 = stringIndexer.transform(data1)

stringIndexer = StringIndexer(inputCol="yr",outputCol = "yrindexed",handleInvalid = 'error').fit(td1)
td2 = stringIndexer.transform(td1)

stringIndexer = StringIndexer(inputCol="mnth",outputCol = "mnthindexed",handleInvalid = 'error').fit(td2)
td3 = stringIndexer.transform(td2)

stringIndexer = StringIndexer(inputCol="hr",outputCol = "hrindexed",handleInvalid = 'error').fit(td3)
td4 = stringIndexer.transform(td3)

stringIndexer = StringIndexer(inputCol="holiday",outputCol = "holidayindexed",handleInvalid = 'error').fit(td4)
td5 = stringIndexer.transform(td4)

stringIndexer = StringIndexer(inputCol="weekday",outputCol = "weekdayindexed",handleInvalid = 'error').fit(td5)
td6 = stringIndexer.transform(td5)

stringIndexer = StringIndexer(inputCol="workingday",outputCol = "workingdayindexed",handleInvalid = 'error').fit(td6)
td7 = stringIndexer.transform(td6)

stringIndexer = StringIndexer(inputCol="weathersit",outputCol = "weathersitindexed",handleInvalid = 'error').fit(td7)
td8 = stringIndexer.transform(td7)

#对前8个类别特征进行OneHot编码


#定义特征向量
featuresArray = ["season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed"]
#把特征组合成特征向量features
assembler = VectorAssembler(inputCols=featuresArray,outputCol="features")
#vecDF = assembler.transform(data1)
vecDF = assembler.transform(log_data)
vecDF.show(3,truncate=False)

+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+------------------------------------------------------+
|season|yr |mnth|hr |holiday|weekday|workingday|weathersit|temp|atemp |hum |windspeed|label |features                                              |
+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+------------------------------------------------------+
|1.0   |0.0|1.0 |0.0|0.0    |6.0    |0.0       |1.0       |0.24|0.2879|0.81|0.0      |2.7726|[1.0,0.0,1.0,0.0,0.0,6.0,0.0,1.0,0.24,0.2879,0.81,0.0]|
|1.0   |0.0|1.0 |1.0|0.0    |6.0    |0.0       |1.0       |0.22|0.2727|0.8 |0.0      |3.6889|[1.0,0.0,1.0,1.0,0.0,6.0,0.0,1.0,0.22,0.2727,0.8,0.0] |
|1.0   |0.0|1.0 |2.0|0.0    |6.0    |0.0       |1.0       |0.22|0.2727|0.8 |0.0      |3.4657|[1.0,0.0,1.0,2.0,0.0,6.0,0.0,1.0,0.22,0.2727,0.8,0.0] |
+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+------+------------------------------------------------------+

featureIndexer = VectorIndexer(inputCol = "features",outputCol="indexedFeatures",maxCategories=24).fit(vecDF)

(trainingData, testData) = vecDF.randomSplit([0.7,0.3])
dt = DecisionTreeRegressor(maxDepth=15,maxBins=64,featuresCol="indexedFeatures")
pipeline = Pipeline(stages=[featureIndexer,dt])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select("prediction","label","features").show(5,truncate=False)

+----------+------+---------------------------------------------------------+
|prediction|label |features                                                 |
+----------+------+---------------------------------------------------------+
|2.5649    |3.091 |[1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.04,0.0758,0.57,0.1045]|
|2.8332    |3.6636|(12,[0,2,7,8,9,10],[1.0,1.0,1.0,0.26,0.303,0.56])        |
|1.6094    |1.6094|[1.0,0.0,1.0,0.0,0.0,1.0,1.0,1.0,0.12,0.1212,0.5,0.2836] |
|1.9459    |2.1972|[1.0,0.0,1.0,0.0,0.0,2.0,1.0,2.0,0.16,0.1364,0.69,0.2836]|
|1.0986    |1.7918|[1.0,0.0,1.0,0.0,0.0,3.0,1.0,1.0,0.2,0.2576,0.64,0.0]    |
+----------+------+---------------------------------------------------------+

evaluator = RegressionEvaluator(labelCol = "label",predictionCol = "prediction",metricName = "mae")
mae = evaluator.evaluate(predictions)
print("Mean Absolute Error on the test data = %g" % mae)

Mean Absolute Error on the test data = 0.331332
