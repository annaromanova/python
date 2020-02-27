
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


# This variable is derived from the number of cores and executors, and will be used to assign the number of model trainers.
num_workers = num_executors * num_processes

print("Number of desired executors: " + 'num_executors')
print("Number of desired processes / executor: " + 'num_processes')
print("Total number of workers: " + 'num_workers')


# In[4]:


#Spark configuration
conf = SparkConf()
conf.set("spark.app.name", application_name)
conf.set("spark.master", master)
conf.set("spark.executor.cores", 'num_processes')
conf.set("spark.executor.instances", 'num_executors')
conf.set("spark.locality.wait", "0")
conf.set("spark.executor.memory", "5g")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");


# In[5]:


#Check that you have working HDFS
get_ipython().system('jps')


# In[6]:


exec(open(os.path.join(os.environ["SPARK_HOME"], 'python/pyspark/shell.py')).read())


# In[7]:


#Check that you have working Spark
get_ipython().system('jps')


# In[8]:


sc = spark.builder.config(conf=conf)             .appName(application_name)             .getOrCreate()


# In[9]:


reader = sc


# In[12]:


rawdata = reader.read.parquet ('hdfs://localhost:9000/user/annaromanova/input/pantheon.tsv.avro.data.parquet') .select("LAT", "LON", "gender","AverageViews")


# In[13]:


rawdata.printSchema()


# In[14]:


print((rawdata.count(), len(rawdata.columns)))


# In[15]:


from pyspark.sql.functions import col, when


# In[16]:


def blank_as_null(x):
    return when(col(x) != "", col(x)).otherwise(None)


# In[17]:


rawdata = rawdata.withColumn("LAT", blank_as_null("LAT"))


# In[18]:


rawdata = rawdata.withColumn("LON", blank_as_null("LON"))


# In[19]:


rawdata = rawdata.withColumn("AverageViews", blank_as_null("AverageViews"))


# In[20]:


rawdata = rawdata.withColumn("gender", blank_as_null("gender"))


# In[21]:


rawdata = rawdata.na.drop(how = 'any')


# In[22]:


print((rawdata.count(), len(rawdata.columns)))


# In[23]:


# Repartition the datasets.
rawdata = rawdata.repartition(num_workers)


# In[24]:


rawdata.show()


# In[25]:


def convertColumn(df, names, newType):
  
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 


# In[26]:


columns = ['LAT', 'LON', 'AverageViews']


# In[27]:


from pyspark.sql.types import FloatType


# In[29]:


rawdata = convertColumn(rawdata, columns, FloatType())


# In[30]:


rawdata.printSchema()


# In[31]:


rawdata = rawdata.select("AverageViews", "LAT", "LON","gender")


# In[32]:


rawdata.take(1)


# In[40]:


from pyspark.ml.feature import OneHotEncoder, StringIndexer


# In[38]:


stringIndexer = StringIndexer(inputCol="gender", outputCol="genderIndex")
model = stringIndexer.fit(rawdata)
indexed = model.transform(rawdata)


# In[41]:


encoder = OneHotEncoder(inputCol="genderIndex", outputCol="genderVec")
encoded = encoder.transform(indexed)
encoded.show()


# In[48]:


encoded = encoded.select("AverageViews", "LAT", "LON", "genderIndex")


# In[49]:


from pyspark.ml.linalg import DenseVector


# In[50]:


# Define the `input_data` 
input_data = encoded.rdd.map(lambda x: (x[0], DenseVector(x[1:])))


# In[51]:


# Replace `df` with the new DataFrame
encoded = spark.createDataFrame(input_data, ["label", "features"])


# In[52]:


encoded.take(1)


# In[53]:


# Split the data into train and test sets
training_set, test_set = encoded.randomSplit([.8,.2],seed=1234)


# In[54]:


from keras import models
from keras import layers


# In[55]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                           input_shape=(2,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))


# In[56]:


optimizer = 'rmsprop'
loss = 'mse'


# In[57]:


model.summary()


# In[58]:


training_set = training_set.repartition(num_workers)
test_set = test_set.repartition(num_workers)
training_set.cache()
test_set.cache()
print("Number of training instances: " + str(training_set.count()))
print("Number of testing instances: " + str(test_set.count()))


# In[59]:


from distkeras.trainers import SingleTrainer


# In[60]:


trainer = SingleTrainer(keras_model=model, loss=loss, worker_optimizer=optimizer, 
                        features_col="features", num_epoch=1, batch_size=64)


# In[61]:


trained_model = trainer.train(training_set)


# In[62]:


# Fetch the training time.
trainer.get_training_time()


# In[63]:


trained_model.get_weights()

