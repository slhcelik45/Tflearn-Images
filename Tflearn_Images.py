import numpy as np
import scipy.ndimage
import tflearn
import tflearn.datasets.mnist as minst

X,Y,testX,testY=minst.load_data(one_hot=True)

imput_layer=tflearn.imput_data(shape=[None,784])
hidden_layer1=tflearn.fully_connected(imput_layer,128,activation='relu',regularizer='L2',weight_decay=0.001)
dropOut1=tflearn.dropout(hidden_layer1,0.08)

hidden_layer2=tflearn.fully_connected(dropOut1,128,activation='relu',regularizer='L2',weight_decay=0.001)
dropOut2=tflearn.dropout(hidden_layer2,0.8)#0.6 ile 0.8 arası yugun olur

softmax=tflearn.fully_connected(dropOut2,10,activation='softmax')
sgd=tflearn.SGD(learning_rate=0.01,lr_decay=0.96,decay_step=1000)
top_K=tflearn.metrics.Top_k(3)
net=tflearn.regression(softmax,optimizer=sgd,metric=top_K,loss='categorical_crossentropy')
model=tflearn.DNN(net,tensorboard_verbose=0)
model.fit(X,Y,n_epoch=10,validation_set=(testX,testY),show_metric=True,run_id='dense_model')
cizim=np.vectorize(lambda x:255-x)(np.ndarray.flatten(scipy.ndimage.imread("resim.png",flatten=True)))
cizim=np.array(cizim).reshape(1,784)
sonuc=model.predict(cizim)
print("a")
print(sonuc)
print("b")




