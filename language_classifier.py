import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf
import DataPreprocess
import random

learning_rate = 0.5

numOfEpochs = 500

dataProcess  = DataPreprocess

# Veriler alınır
germanTrainingSet,englishTrainingSet,germanTestSet,englishTestSet = dataProcess.processData()

#training set ve test set oluşturulur. (Bu DataProcess.py'da da yapılarbilir.)
trainingSet = np.vstack((germanTrainingSet, englishTrainingSet))

testSet = np.vstack((germanTestSet,englishTestSet))

testLabels = np.vstack((np.zeros([200,1]), np.ones([200,1])))

#print(trainingSet)

# Input değerleri için yer açılır. (placeholder)
X = tf.placeholder(tf.float32,[None, 140])

# Ağırlıklar ve bias için değişken tanımlanır.
W = tf.Variable(tf.zeros([140, 2]))
b = tf.Variable(tf.zeros([2]))

# Output değerleri için tensor tanımlanır.
y = tf.nn.softmax(tf.matmul(X, W) + b)

# Doğru label değerleri oluşturulur
y_true = np.vstack((np.zeros([2000,1]), np.ones([2000,1])))

y_ = tf.placeholder(tf.float32, [None, 1])


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices=[1]))

# Gradient Descent kullanılarak cross-entropy minimize edilir
trainStep = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#print(y_true)


#  Training set'e label'lar eklenip shuffle yapılır,
# ardından label sütunu ayrı bir array'a alınır.
trainingSet = np.append(trainingSet, y_true, axis=1)
for i in range(8):
    np.random.shuffle(trainingSet)

y_true = [row[-1] for row in trainingSet]
y_true = np.transpose([y_true])
trainingSet = np.delete(trainingSet, trainingSet[0].__len__() - 1, axis=1)




print("\n\n")
print(y_true.__len__())
print("\n\n")

# Training set'teki verileri sigmoid'den geçirmeyi denedim ama değişiklik olmadı :/
# trainingSet = [[tf.nn.sigmoid(i,name=None) for i in c] for c in trainingSet]

print(trainingSet)
print(y_true)
sess = tf.InteractiveSession()

# Önceden tanımladığımız değişkenleri ilklemek için.
tf.global_variables_initializer().run()


# Training.
for i in range(numOfEpochs):
    sess.run(trainStep, feed_dict={X: trainingSet, y_: y_true})
'''
 Doğruluk oranını bulmak için tahminlerin en büyüğünü
dğru sonucla kıyaslayıp sayıya çevirerek başarı oranı verir.

 Bizde 1.0 çıkıyo çünkü (sanırsam) iki ihitmal de 0.5 olduğu için
hep doğruymuş gibi görünüyo
'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("printin?")
# Modelin tahminlerini yazdırır
print(sess.run(y, feed_dict={X: testSet}))

sess.close()