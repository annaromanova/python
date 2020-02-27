
# coding: utf-8

# In[56]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

import time

import requests

from keras.optimizers import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from pyspark import SparkContext
from pyspark import SparkConf

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from distkeras.trainers import *
from distkeras.predictors import *
from distkeras.transformers import *
from distkeras.evaluators import *
from distkeras.utils import *

from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *


# In[57]:


# Modify these variables according to your needs.
application_name = "Distributed Keras  Analysis"
local = True
if local:
    # Tell master to use local resources.
    master = "local[*]"
    num_processes = 3
    num_executors = 1
else:
    # Tell master to use YARN.
    master = "yarn-client"
    num_executors = 30
    num_processes = 1


# In[58]:


# This variable is derived from the number of cores and executors,
#and will be used to assign the number of model trainers.
num_workers = num_executors * num_processes

print("Number of desired executors: " + 'num_executors')
print("Number of desired processes / executor: " + 'num_processes')
print("Total number of workers: " + 'num_workers')


# In[59]:


#Spark configuration
conf = SparkConf()
conf.set("spark.app.name", application_name)
conf.set("spark.master", master)
conf.set("spark.executor.cores", 'num_processes')
conf.set("spark.executor.instances", 'num_executors')
conf.set("spark.locality.wait", "0")
conf.set("spark.executor.memory", "5g")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");


# In[60]:


#Check that you have working HDFS
get_ipython().system('jps')


# In[61]:


exec(open(os.path.join(os.environ["SPARK_HOME"], 'python/pyspark/shell.py')).read())


# In[62]:


#Check that you have working Spark
get_ipython().system('jps')


# In[63]:


sc = spark.builder.config(conf=conf)             .appName(application_name)             .getOrCreate()


# In[64]:


reader = sc


# In[65]:


rawdata = reader.read.parquet ('hdfs://localhost:9000/user/annaromanova/input/pantheon.tsv.avro.data.parquet') .select("LAT", "LON", "gender","domain")


# In[66]:


rawdata.printSchema()


# In[67]:


print((rawdata.count(), len(rawdata.columns)))


# In[68]:


from pyspark.sql.functions import col, when


# In[69]:


def blank_as_null(x):
    return when(col(x) != "", col(x)).otherwise(None)


# In[70]:


rawdata = rawdata.withColumn("LAT", blank_as_null("LAT"))


# In[71]:


rawdata = rawdata.withColumn("LON", blank_as_null("LON"))


# In[72]:


rawdata = rawdata.withColumn("gender", blank_as_null("gender"))


# In[73]:


rawdata = rawdata.withColumn("domain", blank_as_null("domain"))


# In[74]:


rawdata = rawdata.na.drop(how = 'any')


# In[75]:


print((rawdata.count(), len(rawdata.columns)))


# In[76]:


# Repartition the datasets.
rawdata = rawdata.repartition(num_workers)


# In[77]:


rawdata.show()


# In[78]:


def convertColumn(df, names, newType):
  
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 


# In[79]:


columns = ['LAT', 'LON']


# In[80]:


from pyspark.sql.types import FloatType


# In[81]:


rawdata = convertColumn(rawdata, columns, FloatType())


# In[82]:


rawdata.printSchema()


# In[83]:


rawdata = rawdata.select("domain", "LAT", "LON","gender")


# In[84]:


rawdata.take(1)


# In[85]:


from pyspark.sql.functions import col, countDistinct


# In[86]:


rawdata.agg(*(countDistinct(col(c)).alias(c) for c in rawdata.columns)).show()


# In[87]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer


# In[120]:


stringIndexer = StringIndexer(inputCol="gender", outputCol="genderIndex")
model = stringIndexer.fit(rawdata)
indexed = model.transform(rawdata)


# In[121]:


encoder = OneHotEncoder(inputCol="genderIndex", outputCol="genderVec")
encoded = encoder.transform(indexed)
encoded.show()


# In[122]:


encoded = encoded.select("domain", "LAT", "LON", "genderIndex")


# In[123]:


from pyspark.ml.linalg import DenseVector


# In[124]:


# Define the `input_data` 
input_data = encoded.rdd.map(lambda x: (x[0], DenseVector(x[1:])))


# In[125]:


# Replace `df` with the new DataFrame
encoded = spark.createDataFrame(input_data, ["label", "features"])


# In[126]:


encoded.take(1)


# In[127]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer


# In[128]:


stringIndexer = StringIndexer(inputCol="label", outputCol="label_indexed")
model = stringIndexer.fit(encoded)
indexed = model.transform(encoded)


# In[129]:


encoder = OneHotEncoder(inputCol="label_indexed", outputCol="label_encoded")
encoded = encoder.transform(indexed)
encoded.show()


# In[130]:


encoded.printSchema()


# In[131]:


encoder1 = OneHotTransformer(9, input_col="label_indexed", output_col="label_encod")


# In[132]:


encoded = encoder1.transform(encoded)


# In[133]:


encoded.show()


# In[134]:


# Split the data into train and test sets
training_set, test_set = encoded.randomSplit([.8,.2],seed=1234)


# In[135]:


# Allocate a MinMaxTransformer using Distributed Keras.
# o_min -> original_minimum
# n_min -> new_minimum
transformer = MinMaxTransformer(n_min=0.0, n_max=1.0,                                 o_min=0.0, o_max=250.0,                                 input_col="features",                                 output_col="features_normalized")


# In[136]:


# Transform the dataset.
dataset_train = transformer.transform(training_set)


# In[137]:


dataset_test = transformer.transform(test_set)


# In[138]:


from keras import models
from keras import layers


# In[139]:


reshape_transformer = ReshapeTransformer("features_normalized", "matrix", (1,3))


# In[140]:


dataset_train = reshape_transformer.transform(dataset_train)


# In[141]:


dataset_test = reshape_transformer.transform(dataset_test)


# In[149]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(3,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(9, activation='softmax'))


# In[150]:


model.summary()


# In[151]:


optimizer_mlp = 'adam'
loss_mlp = 'categorical_crossentropy'


# In[145]:


# Assing the training and test set.
training_set = dataset_train.repartition(num_workers)
test_set = dataset_test.repartition(num_workers)
# Cache them.
training_set.cache()
test_set.cache()


# In[146]:


print(training_set.count())


# In[152]:


training_set.show()


# In[160]:


trainer = SingleTrainer(keras_model=model, loss=loss_mlp, worker_optimizer=optimizer_mlp, 
                        features_col="features_normalized", label_col="label_encod", num_epoch=1, batch_size=64)


# In[161]:


trained_model = trainer.train(training_set)


# In[162]:


# Fetch the training time.
trainer.get_training_time()


# In[163]:


trained_model.get_weights()

