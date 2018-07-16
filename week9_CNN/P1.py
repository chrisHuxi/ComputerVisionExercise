# coding=utf-8 
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read the csv file
def openFile(csvname):
    csvpath = './data/'             
    csvfile = csvpath + csvname
    print('Reading "' + csvfile + '":')
    dat = np.loadtxt(csvfile, delimiter=';')
    examples = []
    for row in dat:
        ex = []
        for col in row:
            ex.append(col)
        examples.append(ex)
    return examples  

#training process
#input:  feature, class_label
#output: W ( [2,2] ),b ( [1,2] ) ( for linear classifier )    
def trainPerceptron(feature,cls):
    #batch size: 100 (the whole dataset)
    num_points = cls.shape[0]

    #create computing graph
    graph = tf.Graph()
    with graph.as_default():
        W = tf.Variable(tf.truncated_normal([2,2]))
        b = tf.Variable(tf.zeros([1,2]))
        with tf.name_scope('x_train'):
            x_train = tf.placeholder(tf.float32,shape=(num_points,2))
        with tf.name_scope('y_train'):
            y_train = tf.placeholder(tf.int32, shape=(num_points,1))
            y_train_OneHot = tf.one_hot(y_train, depth = 2)
            #y_train = tf.placeholder(tf.float32, shape=(num_points,1))
        
        y = (tf.matmul(x_train,W) + b) #100,2 + 1,2 = 100,2

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_train_OneHot))

        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
        
    #epoch:
    num_steps = 100000

    #create session
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
            feed_dict = {x_train: feature ,y_train:cls}
            _,_loss = session.run([train,loss],feed_dict = feed_dict)
            if step%5 == 0:
                print(' loss at step %d: %f' %(step,_loss))
        print(W.eval())
        print(b.eval())
        return W.eval(),b.eval()
    
#visualize the result of classifier reference:http://learningtensorflow.com/classifying/
#input:  W ( [2,2] ), b( [1,2] )
#output: image
def visualize(W,b,csvfile,X_values):
    dat = np.loadtxt(csvfile, delimiter=';')
    df = pd.DataFrame(dat)
    
    # Draw graph
    fig, ax = plt.subplots()

    for i, d in enumerate(dat[:,:2]):
        if dat[i,2] < 0:
           plt.plot(d[0], d[1], 'bo')
        else:
            plt.plot(d[0], d[1],color='#FF8000', marker='o')
    
    
    #=======================================================================#        
    #reference : http://learningtensorflow.com/classifying/
    h = 1
    x_min, x_max = X_values[:, 0].min() - 2 * h, X_values[:, 0].max() + 2 * h
    y_min, y_max = X_values[:, 1].min() - 2 * h, X_values[:, 1].max() + 2 * h
    x_0, x_1 = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    decision_points = np.c_[x_0.ravel(), x_1.ravel()]        

    Z = np.argmax(decision_points @ W[[0,1]] + b, axis=1)

    # Create a contour plot of the x_0 and x_1 values
    Z = Z.reshape(int(np.sqrt(Z.shape[0])),int(np.sqrt(Z.shape[0])))
    plt.contourf(x_0, x_1, Z)
    #======================================================================#

    plt.xlim(x_0.min(), x_0.max())
    plt.ylim(x_1.min(), x_1.max())
    plt.title('Distribution (two classes)')
    plt.yticks([]) #Do not show numbers on y-axis
    plt.ylabel('Class 1 = Blue, Class -1 = Orange')
    plt.xlabel('arbitrary values')
    plt.show()

if __name__ == '__main__':
    training_exs = np.array(openFile('data3.csv'))
    feature = training_exs[:,0:2]
    
    cls = training_exs[:,2]
    cls = cls.reshape(cls.shape[0],1)
    W,b = trainPerceptron(feature,cls)
    
    visualize(W,b,'./data/data3.csv',feature)
    