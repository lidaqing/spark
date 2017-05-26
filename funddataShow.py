#coding=utf-8
from pyspark.sql import HiveContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import *

##定义spark
warehouse_location = 'spark-warehouse'

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL Hive integration example") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()
##获取hive中的表，并进行处理 
sqlContext = HiveContext(sc)
sqlContext.sql("use feigu")
df= spark.sql("Select fund_date,nav,accnav from rpt_table_2016113009") 
df1=df.select(substring(trim(df.fund_date),1,7).alias('year'),df.nav,df.accnav)
df2=df1.groupBy("year").mean("nav","accnav").orderBy("year")
##转换为pandas
df3=df2.toPandas()
df4=df3.set_index(['year'])
##画图
df4.plot(kind='bar',rot=30)
## df4.plot(kind='line',rot=30)
