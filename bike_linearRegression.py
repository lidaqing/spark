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
log_data = data1.select(data1.season,data1.yr,data1.mnth,data1.hr,data1.holiday,data1.weekday,data1.workingday,data1.weathersit,data1.temp,data1.atemp,data1.hum,data1.windspeed,round(log(data1.label),4).alias('label'))
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
data2 = OneHotEncoder(inputCol="seasonindexed",outputCol="seasonVec",dropLast=False).transform(td8)
data3 = OneHotEncoder(inputCol="yrindexed",outputCol="yrVec",dropLast=False).transform(data2)
data4 = OneHotEncoder(inputCol="mnthindexed",outputCol="mnthVec",dropLast=False).transform(data3)
data5 = OneHotEncoder(inputCol="hrindexed",outputCol="hrVec",dropLast=False).transform(data4)
data6 = OneHotEncoder(inputCol="holidayindexed",outputCol="holidayVec",dropLast=False).transform(data5)
data7 = OneHotEncoder(inputCol="weekdayindexed",outputCol="weekdayVec",dropLast=False).transform(data6)
data8 = OneHotEncoder(inputCol="workingdayindexed",outputCol="workingdayVec",dropLast=False).transform(data7)
data9= OneHotEncoder(inputCol="weathersitindexed",outputCol="weathersitVec",dropLast=False).transform(data8)

#定义特征向量
featuresArray = ["seasonVec","yrVec","mnthVec","hrVec","holidayVec","weekdayVec","workingdayVec","weathersitVec","temp","atemp","hum","windspeed"]
#把特征组合成特征向量features
assembler = VectorAssembler(inputCols=featuresArray,outputCol="features")
vecDF = assembler.transform(data9)
vecDF.show(3,truncate=False)

+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+-----+-------------+---------+-----------+---------+--------------+--------------+-----------------+-----------------+-------------+-------------+---------------+---------------+-------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------+
|season|yr |mnth|hr |holiday|weekday|workingday|weathersit|temp|atemp |hum |windspeed|label|seasonindexed|yrindexed|mnthindexed|hrindexed|holidayindexed|weekdayindexed|workingdayindexed|weathersitindexed|seasonVec    |yrVec        |mnthVec        |hrVec          |holidayVec   |weekdayVec   |workingdayVec|weathersitVec|features                                                                                |
+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+-----+-------------+---------+-----------+---------+--------------+--------------+-----------------+-----------------+-------------+-------------+---------------+---------------+-------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------+
|1.0   |0.0|1.0 |0.0|0.0    |6.0    |0.0       |1.0       |0.24|0.2879|0.81|0.0      |16.0 |2.0          |1.0      |10.0       |17.0     |0.0           |0.0           |1.0              |0.0              |(4,[2],[1.0])|(2,[1],[1.0])|(12,[10],[1.0])|(24,[17],[1.0])|(2,[0],[1.0])|(7,[0],[1.0])|(2,[1],[1.0])|(4,[0],[1.0])|(61,[2,5,16,35,42,44,52,53,57,58,59],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.24,0.2879,0.81])|
|1.0   |0.0|1.0 |1.0|0.0    |6.0    |0.0       |1.0       |0.22|0.2727|0.8 |0.0      |40.0 |2.0          |1.0      |10.0       |19.0     |0.0           |0.0           |1.0              |0.0              |(4,[2],[1.0])|(2,[1],[1.0])|(12,[10],[1.0])|(24,[19],[1.0])|(2,[0],[1.0])|(7,[0],[1.0])|(2,[1],[1.0])|(4,[0],[1.0])|(61,[2,5,16,37,42,44,52,53,57,58,59],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.22,0.2727,0.8]) |
|1.0   |0.0|1.0 |2.0|0.0    |6.0    |0.0       |1.0       |0.22|0.2727|0.8 |0.0      |32.0 |2.0          |1.0      |10.0       |21.0     |0.0           |0.0           |1.0              |0.0              |(4,[2],[1.0])|(2,[1],[1.0])|(12,[10],[1.0])|(24,[21],[1.0])|(2,[0],[1.0])|(7,[0],[1.0])|(2,[1],[1.0])|(4,[0],[1.0])|(61,[2,5,16,39,42,44,52,53,57,58,59],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.22,0.2727,0.8]) |
+------+---+----+---+-------+-------+----------+----------+----+------+----+---------+-----+-------------+---------+-----------+---------+--------------+--------------+-----------------+-----------------+-------------+-------------+---------------+---------------+-------------+-------------+-------------+-------------+------------------------------------------------------------------------

lr = LinearRegression(featuresCol="features",labelCol="label",fitIntercept=True,maxIter=20,regParam=0.3,elasticNetParam=0.8)
lrModel = lr.fit(vecDF)
predictions = lrModel.transform(vecDF)
trainingSummary = lrModel.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)

print("RMSE: %f" % trainingSummary.rootMeanSquaredError)

print("r2: %f" % trainingSummary.r2)
r2: 0.686258


#计算RMSLE评估指标
predictions.createOrReplaceTempView("predictDF")
rmsle = spark.sql("select sqrt(sum((log(prediction+1)-log(label+1))*(log(prediction+1)-log(label+1)))/count(1)) as diff from predictDF")
rmsle.show()

+------------------+
|              diff|
+------------------+
|0.8053506104776464|
+------------------+
