from pyspark.sql import SparkSession
from pyspark.sql import Row

spark = SparkSession \
      .builder \
      .appName("Python Spark SQL basic example") \
      .config("spark.some.config.option","some-value") \
      .getOrCreate()
      
sc = spark.sparkContext

userrdd = sc.textFile("hdfs://master:9000/home/hadoop/data/u.user").map(lambda line: line.split("|"))
df = userrdd.map(lambda fields:Row(name=fields[0],age=int(fields[1]),gender=fields[2],occupation=fields[3],zip=fields[4]))

schemauser = spark.createDataFrame(df)
schemauser.createOrReplaceTempView("user")

age = spark.sql("SELECT age FROM user")
age.show(5)

ages = age.rdd.map(lambda p:p.age).collect()
hist(ages,bins=20,color='lightblue',normed=True)

count_occp = spark.sql("SELECT occupation,count(occupation) as cnt FROM user Group by occupation order by cnt")

count_occp.show(5)

x_axis = count_occp.rdd.map(lambda p:p.occupation).collect()
y_axis = count_occp.rdd.map(lambda p:p.cnt).collect()

pos = np.arange(len(x_axis))
width = 1.0

ax = plt.axes()
ax.set_xticks(pos+(width/2))
ax.set_xticklabels(x_axis)
plt.bar(pos,y_axis,width,color='orange')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(16,10)
