from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import tensorflow as tf


# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1)) # channel is the last element in tensorflow
y = tf.placeholder(tf.float32, shape=(None, 10))


# Define TF model graph
from cleverhans.utils_keras import cnn_model
model = cnn_model()
predictions = model(x)


# Get MNIST data
from cleverhans.utils_mnist import data_mnist
X_train, Y_train, X_test, Y_test = data_mnist()


# Train an MNIST model
from cleverhans.utils_tf import model_train, model_eval
sess = tf.Session()
train_params = {
    'nb_epochs': 6,
    'batch_size': 128,
    'learning_rate': 0.001,
}
model_train(sess, x, y, predictions, X_train, Y_train, args=train_params)


# Evaluate the MNIST model
eval_params = {'batch_size': 128}
accuracy = model_eval(sess, x, y, predictions, X_test, Y_test, args=eval_params)
print('Test accuracy on legitimate test examples: ' + str(accuracy))
# Test accuracy on legitimate test examples: 0.9888


# Craft adversarial examples using the Fast Gradient Sign Method
from cleverhans.attacks_tf import fgsm
from cleverhans.utils_tf import batch_eval
adv_x = fgsm(x, predictions, eps=0.3)
X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test, args=eval_params)
print('Test accuracy on adversarial examples: ' + str(accuracy))
# Test accuracy on adversarial examples: 0.0837


# Adversarial training
model_2 = cnn_model()
predictions_2 = model_2(x)
adv_x_2 = fgsm(x, predictions_2, eps=0.3)
predictions_2_adv = model_2(adv_x_2)
model_train(sess, x, y, predictions_2, X_train, Y_train, predictions_adv=predictions_2_adv, args=train_params)


# Evaluate the accuracy on legitimate examples
accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test, args=eval_param)
print('Test accuracy on legitimate test examples: ' + str(accuracy))
# Test accuracy on legitimate test examples: 0.9897


# Evaluate the accuracy on adversarial examples
X_test_adv_2, = batch_eval(sess, [x], [adv_x_2], [X_test], args=eval_params)
accuracy_adv = model_eval(sess, x, y, predictions_2, X_test_adv_2, Y_test, args=eval_params)
print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
# Test accuracy on adversarial examples: 0.9411

