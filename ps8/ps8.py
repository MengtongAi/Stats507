import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
sc=SparkContext()
sc
sql=SQLContext(sc)
df=pd.read_csv("triangles.csv")
df=pd.read_csv("/user/jbhender/stats507/triangles.csv")
df=pd.read_csv("hdfs://user/jbhender/stats507/triangles.csv")
df=sc.textFile("hdfs://user/jbhender/stats507/triangles.csv")
cols=df.map(lambda x:x.split(','))
cols
col.show()
cols.show()
df.head()
cols.collect()
df = spark.read.csv(r"hdfs://user/jbhender/stats507/triangles.csv", encoding='utf-8', header=True, inferSchema=True)
q
exit()
df = spark.read.csv(r"hdfs://user/jbhender/stats507/triangles.csv", encoding='utf-8', header=True, inferSchema=True)
df = spark.read.csv(r"hdfs:/user/jbhender/stats507/triangles.csv", encoding='utf-8', header=True, inferSchema=True)
df.show()
df.count()
df['base']
df2=spark.read.csv(r"hdfs:/user/jbhender/stats507/rectangles.csv", encoding='utf-8', header=True, inferSchema=True)
df2.show()
count1=df.count()
count2=df.count()
count1
count2
area1=df['base']*df['height']
area1=area1.sum()
from pyspark.sql.functions import udf
compute_Area=udf(lambda base,height:base*height/2)
df1_area=df.withColumn("area",compute_Area(df.base,df.height))
df1_area.show()
df1_area.area.sum()
res=[[]]
res[0].append(count1)
res[1].append(count1)
res.append([])
res[1].append(count2)
sum1=df1_area.agg({"area":"sum"}).collect()[0]
sum1=sum1["sum(area)"]
sum1
res[0].append(sum1)
df1_area["mean(area)"]
mean1=df1_area.agg({"area":"mean"}).collect()[0]
mean1=mean1['mean(area)']
mean1=mean1["mean(area)"]
mean1.show()
mean
mean1
mean1['avg(area)']
res[0].append(mean1['avg(area)'])
df2_area=df2.withColumn("area",compute_Area(df2.base,df2.height))
df2
compute_Area2=udf(lambda w,l:w*l)
df2_area=df2.withColumn("area",compute_Area2(df2.width,df2.length))
sum2=df2_area.agg({"area":"sum"}).collect()[0]
sum2=sum2['sum(area)']
sum2
res[1].append(SUM2)
res[1].append(sum2)
mean2=df2_area.agg({"area":"mean"}).collect()[0]
mean2.show()
mean2
mean2=mean2['avg(area)']
res[1].append(mean2)
import pandas as pd
pd.Dataframes(res)
pd.Dataframe(res)
res_final=pd.DataFrame(res)
res_final
res_final.columns=['count','SUM of area',"mean of area"]
res_final.to_csv("res.csv",sep=",")
import readline
readline.write_history_file("ps8.py")
