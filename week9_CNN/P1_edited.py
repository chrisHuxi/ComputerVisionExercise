# coding=utf-8 
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    
def trainPerceptron(feature,cls):
    num_points = cls.shape[0]
    graph = tf.Graph()
    
    with graph.as_default():
        W = tf.Variable(tf.truncated_normal([2,1]))
        b = tf.Variable(tf.zeros([1,1]))
        with tf.name_scope('x_train'):
            x_train = tf.placeholder(tf.float32,shape=(num_points,2))
        with tf.name_scope('y_train'):
            y_train = tf.placeholder(tf.float32, shape=(num_points,1))
        
        y = tf.nn.softsign(tf.matmul(x_train,W) + b) #100,2 + 1,2 = 100,2

        loss = tf.reduce_mean(tf.square(y-y_train))
        
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        train = optimizer.minimize(loss)
        
    num_steps = 1000


    #创建会话:通常来说这一块相对固定
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
    
def visualize(W,b,csvfile):
    
    print('Reading "' + csvfile + '":')
    dat = np.loadtxt(csvfile, delimiter=';')
    # print(dat)

    print('\n')

    df = pd.DataFrame(dat)
    print(df.describe())

    # Draw graph
    fig, ax = plt.subplots()

    for i, d in enumerate(dat[:,:2]):
        if dat[i,2] < 0:
           plt.plot(d[0], d[1], 'bo')
        else:
            plt.plot(d[0], d[1],color='#FF8000', marker='o')
    
   
    x = np.linspace(-1, 5, 10)
    y = (- b -x * W[0][0])/W[1][0]
    y = y.flatten()
    
    plt.plot(x, y, color="red")
    
    plt.title('Distribution (two classes)')
    plt.yticks([]) #Do not show numbers on y-axis
    plt.ylabel('Class 1 = Blue, Class -1 = Orange')
    plt.xlabel('arbitrary values')
    plt.show()

if __name__ == '__main__':
    training_exs = np.array(openFile('data3.csv'))
    feature = training_exs[:,0:2]
    
    cls = training_exs[:,2]
    print(cls)
    cls = cls.reshape(cls.shape[0],1)
    W,b = trainPerceptron(feature,cls)
    
    visualize(W,b,'./data/data3.csv')
    