from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder, StringIndexer

spark = SparkSession \
      .builder \
      .appName("Python Spark SQL basic example") \
      .config("spark.some.config.option","some-value") \
      .getOrCreate()
     
userdf = spark.sql("SELECT age,gender,occupation FROM user")

Indexer = StringIndexer(inputCol="occupation",outputCol="occupindex")
model = Indexer.fit(userdf)
indexed = model.transform(userdf)

encoder = OneHotEncoder(inputCol="occupindex",outputCol="occupVec")
encoded = encoder.transform(indexed)

Indexer = StringIndexer(inputCol="gender",outputCol="genderindex")
model = Indexer.fit(encoded)
indexed = model.transform(encoded)

encoder = OneHotEncoder(inputCol="genderindex",outputCol="genderVec")
encoded1 = encoder.transform(indexed)

#assembler = VectorAssembler(inputCols(Array("name","age","gender","occupVec")),outputCol("features"))
vecAssembler = VectorAssembler(inputCols=["age","genderVec","occupVec"],outputCol="features")

vecDF = vecAssembler.transform(encoded1)
vecDF.select("features").show(10)
