
# coding: utf-8

# In[57]:


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


# In[33]:


rawdata = reader.read.parquet ('hdfs://localhost:9000/user/annaromanova/input/pantheon.tsv.avro.data.parquet') .select("LAT", "LON", "AverageViews")


# In[34]:


rawdata.printSchema()


# In[35]:


print((rawdata.count(), len(rawdata.columns)))


# In[36]:


from pyspark.sql.functions import col, when


# In[37]:


def blank_as_null(x):
    return when(col(x) != "", col(x)).otherwise(None)


# In[38]:


rawdata = rawdata.withColumn("LAT", blank_as_null("LAT"))


# In[39]:


rawdata = rawdata.withColumn("LON", blank_as_null("LON"))


# In[40]:


rawdata = rawdata.withColumn("AverageViews", blank_as_null("AverageViews"))


# In[41]:


rawdata = rawdata.na.drop(how = 'any')


# In[42]:


print((rawdata.count(), len(rawdata.columns)))


# In[43]:


# Repartition the datasets.
rawdata = rawdata.repartition(num_workers)


# In[44]:


rawdata.show()


# In[45]:


def convertColumn(df, names, newType):
  
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 


# In[46]:


columns = ['LAT', 'LON', 'AverageViews']


# In[47]:


from pyspark.sql.types import FloatType


# In[48]:


rawdata = convertColumn(rawdata, columns, FloatType())


# In[49]:


rawdata.printSchema()


# In[51]:


rawdata = rawdata.select("AverageViews", "LAT", "LON")


# In[52]:


rawdata.take(1)


# In[53]:


from pyspark.ml.linalg import DenseVector


# In[54]:


# Define the `input_data` 
input_data = rawdata.rdd.map(lambda x: (x[0], DenseVector(x[1:])))


# In[55]:


# Replace `df` with the new DataFrame
rawdata = spark.createDataFrame(input_data, ["label", "features"])


# In[56]:


rawdata.take(1)


# In[88]:


# Split the data into train and test sets
training_set, test_set = rawdata.randomSplit([.8,.2],seed=1234)


# In[89]:


from keras import models
from keras import layers


# In[90]:


model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                           input_shape=(2,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))


# In[91]:


optimizer = 'rmsprop'
loss = 'mse'


# In[92]:


model.summary()


# In[93]:


training_set = training_set.repartition(num_workers)
test_set = test_set.repartition(num_workers)
training_set.cache()
test_set.cache()
print("Number of training instances: " + str(training_set.count()))
print("Number of testing instances: " + str(test_set.count()))


# In[94]:


from distkeras.trainers import SingleTrainer


# In[95]:


trainer = SingleTrainer(keras_model=model, loss=loss, worker_optimizer=optimizer, 
                        features_col="features", num_epoch=1, batch_size=64)


# In[96]:


trained_model = trainer.train(training_set)


# In[97]:


# Fetch the training time.
trainer.get_training_time()


# In[98]:


trained_model.get_weights()

