import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf

epochs = 1000

dataFrame = pd.read_csv('Pokemon.csv')


# dataFrame = dataFrame.drop(['Poison','#'])

trainingSize = (int)(dataFrame.__len__() * 0.75)

dataSet = dataFrame.drop(['#', 'Type 2'], axis=1)

colName = dataFrame.coloumns[1]


dataFrame = dataFrame[dataFrame.Type1 == 'Fire' or dataFrame.Type1 == 'Water' or
dataFrame.Type1 == 'Normal' or dataFrame.Type1 == 'Grass' or dataFrame.Type1 == 'Poison']

trainingSet = dataSet[0:trainingSize]
print(trainingSet)

x = tf.placeholder(tf.int16, [None,600])

weights = tf.Variable(tf.zeros([600,18]))

b = tf.Variable(tf.zeros([18]))

y = tf.nn.softmax(tf.matmul(x, weights) + b)

y_ = tf.placeholder(tf.int16, [None, 18])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

# for i in range(epochs):



# 18 Types of Pokemon
#f = tf.contrib.learn.

