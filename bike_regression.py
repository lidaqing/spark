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
log_data = data1.select(data1.season,data1,yr,data1.mnth,data1.hr,data1.holiday,data1.weekday,data1.workingday,data1.weathersit,\
data1.temp,data1.atemp,data1.hum,data1.windspeed,round(log(data1.label),4).alias('label'))
log_data.show(2)

#把前8个分类特征索引化
stringIndexer = StringIndexer(inputCol = "season",outputCol = "seasonindexed",handleInvalid = 'error').fit(data1)
td1 = stringIndex.transform(data1)

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

featureIndexer = VectorIndexer(inputCol = "features",outputCol="indexedFeatures",maxCategories=24).fit(vecDF)

(trainingData, testData) = vecDF.randomSplit([0.7,0.3])
dt = DecisionTreeRegressor(maxDepth=15,maxBins=64,featuresCol="indexedFeatures")
pipeline = Pipeline(stages=[featureIndexer,dt])
model = pipeline.fit(trainingData)
predictions = model.transfrom(testData)
predictions.select("prediction","label","features").show(5,truncate=False)
evaluator = RegressionEvaluator(labelCol = "label",predictionCol = "prediction",metricName = "mae")
mae = evaluator.evaluate(predictions)
print("Mean Absolute Error on the test data = %g" % mae)
