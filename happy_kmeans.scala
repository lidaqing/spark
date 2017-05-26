//spark-shell --master spark://master:7077 --driver-memory 1G --total-executor-cores 2

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.DataFrameStatFunctions
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.clustering.KMeans
import spark.implicits._

val data = spark.read.format("csv").option("header",true).load("hdfs://master:9000/home/hadoop/data/happy01.csv")
data.cache()

//把字段类型转换为double类型
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

data1.createOrReplaceTempView("data")
val education_ch = "ifnull(education,14) as education"
val dataDF = spark.sql(s"select affairs,gender,age,yearsmarried,children,religiousness,$education_ch,occupation,rating from data")

//把特征gender字符转换成数字索引
val indexer = new StringIndexer().setInputCol("gender").setOutputCol("genderIndex").fit(dataDF)
val indexed = indexer.transform(dataDF)

//OneHot编码，注意setDropLast设置为false
val encoder = new OneHotEncoder().setInputCol("genderIndex").setOutputCol("genderVec").setDropLast(false)
val encoded = encoder.transform(indexed)

//对特征或字段children做相同操作
val indexer1 = new StringIndexer().setInputCol("children").setOutputCol("childrenIndex").fit(encoded)
val indexed1 = indexer1.transform(encoded)
val encoder1 = new OneHotEncoder().setInputCol("childrenIndex").setOutputCol("childrenVec").setDropLast(false)
val encoded1 = encoder1.transform(indexed1)
val encodeDF = encoded1

//将字段组合成向量feature
val assembler = new VectorAssembler().setInputCols(Array("affairs","age","yearsmarried","religiousness","education","occupation","rating","genderVec","childrenVec")).setOutputCol("features")
val vecDF = assembler.transform(encodeDF)

//标准化 ，均值标准差
val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(true)
val scalerModel = scaler.fit(vecDF)
val scaledData:DataFrame = scalerModel.transform(vecDF)

//主成分
val pca = new PCA().setInputCol("scaledFeatures").setOutputCol("pcaFeatures").setK(5).fit(scaledData)

//解释变量方差
pca.explainedVariance.values

res14: Array[Double] = Array(0.28784096351534705, 0.23738839271940232, 0.11742666151965797, 0.09270458337365094, 0.08418713886564144)

//载荷
pca.pc

res15: org.apache.spark.ml.linalg.DenseMatrix =
-0.12039252794389527  0.051407830997346096  ... (5 total)
-0.4284798005502755   0.05416317873920609   ...
-0.4441537879481735   0.19266852851394048   ...
-0.12220299906148521  0.0811847906977086    ...
-0.14697860994797401  -0.38430434806965824  ...
-0.14485607125868386  -0.43071086298928507  ...
0.17706662969316903   -0.12800219539673793  ...
0.2452811292679256    0.4930474062682183    ...
-0.2452811292679256   -0.4930474062682183   ...
-0.44465733444069466  0.23972246634015676   ...
0.4446573344406945    -0.2397224663401567   ...

pca.extractParamMap()

res18: org.apache.spark.ml.param.ParamMap =
{
	pca_960b16064aa1-inputCol: scaledFeatures,
	pca_960b16064aa1-k: 5,
	pca_960b16064aa1-outputCol: pcaFeatures
}


pca.params

res19: Array[org.apache.spark.ml.param.Param[_]] = Array(pca_960b16064aa1__inputCol, pca_960b16064aa1__k, pca_960b16064aa1__outputCol)

val pcaDF:DataFrame = pca.transform(scaledData)
pcaDF.cache()

//注意最大迭代次数和轮廓系数
val KSSE = (2 to 20 by 1).toList.map{k => 
    val kmeans = new KMeans().setK(k).setSeed(123).setFeaturesCol("scaledFeatures")
    val model = kmeans.fit(scaledData)
    val WSSSE = model.computeCost(scaledData)
    (k,model.getMaxIter,WSSSE,model.summary.cluster,model.summary.clusterSizes,model.clusterCenters)    
}

//根据SSE确定K值
val KSSEdf:DataFrame=KSSE.map{x => (x._1,x._2,x._3,x._5)}.toDF("K","MaxIter","SSE","clusterSizes")
KSSE.foreach(println)
