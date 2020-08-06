import numpy as np
import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs['acc']>0.98):
          print("\nReached 98% accuracy so cancelling training 😁")
          self.model.stop_training = True

class cnn:
   def __init__(self,train_x,train_y,test_x,test_y):
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.test_x = np.array(test_x)
        self.test_y = np.array(test_y)
        self.train_x = self.train_x/255
        self.test_x = self.test_x/255
        self.model = tf.keras.Sequential([
                      tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)),
                      tf.keras.layers.MaxPool2D(2,2),
                      tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                      tf.keras.layers.MaxPool2D(2,2),
                      tf.keras.layers.Conv2D(128,(3,3),activation="relu"),
                      tf.keras.layers.MaxPool2D(2,2),
                      tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(256,activation="relu"),
                      tf.keras.layers.Dense(6,activation="softmax")
                      ])

   def print_model_summary(self):
    print("model summary 👀")
    self.model.summary()
    
   def train_model(self,epochs,batch_size):
        print("start traing 🐱‍👤 ...")
        self.model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
        callback = Callback()
        self.model.fit(self.train_x,self.train_y,epochs=epochs,batch_size=batch_size,callbacks=[callback])
        print("finshed traing 🚀🚀 .")
        
   def evaluate_model(self): 
        print("Evaluate on test data 🤔")
        results = self.model.evaluate(self.test_x,self.test_y,batch_size=1)
        print("test accuracy",round(results[1]*100),sep=" : ",end=" % \n")
        
   def save_model(self,path): self.model.save(path)
            


