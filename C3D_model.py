import tensorflow as tf
import numpy as np 
import os
from params import PARAMS

layers = tf.keras.layers

class C3D(tf.keras.Model):
    def __init__(self):
        super(C3D, self).__init__()
        init = tf.glorot_normal_initializer()

        # conv1
        self.conv1 = layers.Conv3D(filters=64, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv1")
        self.pool1 = layers.MaxPool3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name="pool1")

        # conv2
        self.conv2 = layers.Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv2")
        self.pool2 = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name="pool2")
        
        # conv3
        self.conv3a = layers.Conv3D(filters=256, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv3a")
        self.conv3b = layers.Conv3D(filters=256, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv3b")
        self.pool3 = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name="pool3")
        
        # conv4
        self.conv4a = layers.Conv3D(filters=512, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv4a")
        self.conv4b = layers.Conv3D(filters=512, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv4b")
        self.pool4 = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name="pool4")
        
        # conv4
        self.conv5a = layers.Conv3D(filters=512, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv5a")
        self.conv5b = layers.Conv3D(filters=512, kernel_size=(3,3,3), strides=(1,1,1), padding='same', 
                                   data_format='channels_last', kernel_initializer=init, name="conv5b")
        self.pool5 = layers.MaxPool3D(pool_size=(2,2,2), strides=(2,2,2), padding='same', name="pool5")

        self.avgPool = layers.GlobalAveragePooling3D()
        self.logits = layers.Dense(PARAMS['num_classes'], activation=None, name="logits")


    def predict(self, x):
        # input: (10, 16, 112, 112, 3)
        x = self.conv1(x)  # (10, 16, 112, 112, 64)
        x = tf.nn.relu(x)
        x = self.pool1(x)  # (10, 16, 56, 56, 64)

        x = self.conv2(x)  # (10, 16, 56, 56, 128)
        x = tf.nn.relu(x)
        x = self.pool2(x)  # (10, 8, 28, 28, 128)

        x = self.conv3a(x)  # (10, 8, 28, 28, 256)
        x = tf.nn.relu(x)
        x = self.conv3b(x)  # (10, 8, 28, 28, 256)
        x = tf.nn.relu(x)
        x = self.pool3(x)  # (10, 4, 14, 14, 256)

        x = self.conv4a(x)  # (10, 4, 14, 14, 512) 
        x = tf.nn.relu(x)
        x = self.conv4b(x)  # (10, 4, 14, 14, 512)
        x = tf.nn.relu(x)
        x = self.pool4(x)   # (10, 2, 7, 7, 512)

        x = self.conv5a(x)  # (10, 2, 7, 7, 512)
        x = tf.nn.relu(x)
        x = self.conv5b(x)  # (10, 2, 7, 7, 512)
        x = tf.nn.relu(x)
        x = self.pool5(x)   # (10, 1, 4, 4, 512)

        x = self.avgPool(x)  # (10, 512)
        y = self.logits(x)   # (10, 101)
        return y


    def loss_fn(self, X, y):
        w = PARAMS["reg_w"]
        logits = self.predict(X)
        loss_ent = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_variables]) * w
        loss = loss_ent + loss_reg
        return loss, loss_ent, loss_reg

    
    def acc_fn(self, X, y):
        preds = self.predict(X).numpy()
        acc = np.sum(np.argmax(preds, axis=1) == y.numpy(), dtype=np.float32) / X.numpy().shape[0]
        return acc


# if __name__ == '__main__':
#     model = C3D()
#     x = tf.convert_to_tensor(np.random.random((10, PARAMS['num_frames_per_clip'], 112, 112, 3)), dtype=tf.float32)
#     y = model.predict(x)
#     print(y.shape)

    # total parameters
    # for v in model.trainable_variables:
    #     print(v.name, v.get_shape())

    # total_para = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    # print("total parameters:", total_para)   # 27740133

#     y_ = np.random.randint(0, 101, 10)
#     loss, loss_ent, loss_reg = model.loss_fn(x, y_)
#     print("loss:", loss.numpy(), loss_ent.numpy(), loss_reg.numpy())