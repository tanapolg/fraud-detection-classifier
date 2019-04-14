import pandas as pd
import numpy as np

# Import and store dataset
credit_card_data = pd.read_csv('creditcard.csv')
#print(credit_card_data)

#my
class_names = credit_card_data['Class']
# To separate our data into few sets (training etc):
# Splitting data into 4 sets:
# 1. Shuffle/randomize data
# 2. One-hot encoding
# 3. Normalize our data (we'll convert negative data
# into numbers between 0 and 1)
# 4. Splitting up X/y values (x values - our inputs,
# y values - out outputs; x - columns from 1 till 28 in our .csv,
# y - one-hot encodings - 0..1 - will tell us our transactions are fraudulent or not)
# 5. Convert data_frames (creadit_card_data from .csv)
# to numpy arrays (float32 - TensorFlow loves to work with float32)
# 6. Splitting the final data into X/y train/test - so we'll have 4 arrays:
# X train, y train, X test, y test

# Shuffle and randomize data
shuffled_data = credit_card_data.sample(frac=1)
# Change Class column into Class_0 ([1 0] for legit data) and Class_1 ([0 1] for fraudulent data)
one_hot_data = pd.get_dummies(shuffled_data, columns=['Class'])
# Change all values into numbers between 0 and 1
normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())
# Store just columns V1 through V28 in df_X and columns Class_0 and Class_1 in df_y
df_X = normalized_data.drop(['Class_0', 'Class_1'], axis=1)
df_y = normalized_data[['Class_0', 'Class_1']]
# Convert both data_frames into np arrays of float32
ar_X, ar_y = np.asarray(df_X.values, dtype='float32'), np.asarray(df_y.values, dtype='float32')
# Allocate first 80% of data into training data and remaining 20% into testing data
train_size = int(0.8 * len(ar_X))
(raw_X_train, raw_y_train) = (ar_X[:train_size], ar_y[:train_size])
(raw_X_test, raw_y_test) = (ar_X[train_size:], ar_y[train_size:])

# Gets a percent of fraud vs legit transactions (0.0017% of transactions are fraudulent)
count_legit, count_fraud = np.unique(credit_card_data['Class'], return_counts=True)[1]
fraud_ratio = float(count_fraud / (count_legit + count_fraud))
print('Percent of fraudulent transactions: ', fraud_ratio)

# Applies a logit weighting of 578 (1/0.0017) to fraudulent transactions to cause model to pay more attention to them
weighting = 1 / fraud_ratio
raw_y_train[:, 1] = raw_y_train[:, 1] * weighting

import tensorflow as tf

# 30 cells for the input
input_dimensions = ar_X.shape[1]
# 2 cells for the output
output_dimensions = ar_y.shape[1]
# 100 cells for the 1st layer
num_layer_1_cells = 100
# 150 cells for the second layer
num_layer_2_cells = 150

# We will use these as inputs to the model when it comes time to train it (assign values at run time)
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='y_train')

# We will use these as inputs to the model once it comes time to test it
X_test_node = tf.constant(raw_X_test, name='X_test')
y_test_node = tf.constant(raw_y_test, name='y_test')

# First layer takes in input and passes output to 2nd layer
weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name='weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name='biases_1')

# Second layer takes in input from 1st layer and passes output to 3rd layer
weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]), name='weight_2')
biases_2_node = tf.Variable(tf.zeros([num_layer_2_cells]), name='biases_2')

# Third layer takes in input from 2nd layer and outputs [1 0] or [0 1] depending on fraud vs legit
weight_3_node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]), name='weight_3')
biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name='biases_3')


# Function to run an input tensor through the 3 layers and output a tensor that will give us a fraud/legit result
# Each layer uses a different function to fit lines through the data and predict whether a given input tensor will \
#   result in a fraudulent or legitimate transaction
def network(input_tensor):
    # Sigmoid fits modified data well
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
    # Dropout prevents model from becoming lazy and over confident
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_node) + biases_2_node), 0.85)
    # Softmax works very well with one hot encoding which is how results are outputted
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3_node) + biases_3_node)
    return layer3


# Used to predict what results will be given training or testing input data
# Remember, X_train_node is just a placeholder for now. We will enter values at run time
y_train_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)

# Cross entropy loss function measures differences between actual output and predicted output
cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_prediction)

# Adam optimizer function will try to minimize loss (cross_entropy) but changing the 3 layers' variable values at a
#   learning rate of 0.005
optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


from sklearn.metrics import confusion_matrix
from sklearn import metrics
from ggplot import *
# Function to calculate the accuracy of the actual result vs the predicted result
def calculate_accuracy(actual, predicted, buildMatrix):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    # my confusion matrix
    if buildMatrix:
        a = 0
        f = 0
        for i in actual:
            if actual[i] == 0:
                a = a + 1
            if actual[i] == 1:
                f = f + 1
        print('actual: ')
        print(actual.size)
        print(a)
        print(f)
        ap = 0
        fp = 0
        for i in predicted:
            if predicted[i] == 0:
                ap = ap + 1
            else:
                fp = fp + 1
        print('predicted: ')
        print(predicted.size)
        print(ap)
        print(fp)

        fpr, tpr, threshold = metrics.roc_curve(actual, predicted)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        # method II: ggplot
        df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
        ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
        # Compute confusion matrix
        # cnf_matrix = confusion_matrix(actual, predicted)
        # np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                       title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        #                       title='Normalized confusion matrix')
        # plt.interactive(False)
        # plt.show(block=True)
    return (100 * np.sum(np.equal(predicted, actual)) / predicted.shape[0])

num_epochs = 100

import time

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):

        start_time = time.time()

        _, cross_entropy_score = session.run([optimizer, cross_entropy],
                                             feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})

        if epoch % 10 == 0:
            timer = time.time() - start_time

            print('Epoch: {}'.format(epoch), 'Current loss: {0:.4f}'.format(cross_entropy_score),
                  'Elapsed time: {0:.2f} seconds'.format(timer))

            final_y_test = y_test_node.eval()
            final_y_test_prediction = y_test_prediction.eval()
            final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction, False)
            print("Current accuracy: {0:.2f}%".format(final_accuracy))

    final_y_test = y_test_node.eval()
    final_y_test_prediction = y_test_prediction.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction, False)
    print("Final accuracy: {0:.2f}%".format(final_accuracy))

final_fraud_y_test = final_y_test[final_y_test[:, 1] == 1]
final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:, 1] == 1]
final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction, True)
print('Final fraud specific accuracy: {0:.2f}%'.format(final_fraud_accuracy))
