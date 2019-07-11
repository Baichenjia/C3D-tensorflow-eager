"""Trains and Evaluates the MNIST network using a feed dictionary."""
import os
import time
import tensorflow as tf
import input_data
from C3D_model import C3D
from params import PARAMS
import math
import numpy as np
from input_data import read_clip_and_label

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution(config=config)

def train(model, optimizer, save_dir, training_step):
    loss_train_log, loss_val_log, acc_log = [], [], []
    acc_best = 0.0
    for step in range(training_step):
        train_images, train_labels, _, _ = read_clip_and_label(filename='list/train.list')
        X, y = tf.convert_to_tensor(train_images/255., tf.float32), tf.convert_to_tensor(train_labels, tf.int64)    # (10, 16, 112, 112, 3)  (10,)

        with tf.GradientTape() as tape:
            loss, loss_ent, loss_reg = model.loss_fn(X, y)
        acc_train = model.acc_fn(X, y)
        print(loss.numpy(), loss_ent.numpy(), loss_reg.numpy(), acc_train*100, "%")
        loss_train_log.append([loss.numpy(), loss_ent.numpy(), loss_reg.numpy()])

        # 使用 loss 训练
        gradients = tape.gradient(loss, model.trainable_variables)
        # gradients, n = tf.clip_by_global_norm(gradients, 1.)
        # print("grad norm:", n.numpy())
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # validation
        if step % 50 == 0:
            acc_tmp, loss_tmp, loss_ent_tmp, loss_reg_tmp = [], [], [], []
            for i in range(5):
                val_images, val_labels, _, _ = read_clip_and_label(filename='list/test.list')
                X_val, y_val = tf.convert_to_tensor(val_images/255., tf.float32), tf.convert_to_tensor(val_labels, tf.int64)
                loss_val, loss_ent_val, loss_reg_val = model.loss_fn(X_val, y_val)
                acc = model.acc_fn(X_val, y_val)
                acc_tmp.append(acc)
                loss_tmp.append(loss_val.numpy())
                loss_ent_tmp.append(loss_ent_val.numpy())
                loss_reg_tmp.append(loss_reg_val.numpy())

            print("\nTEST------------")
            print("step=", step)
            print("val loss:", np.mean(loss_tmp), "\tloss ent", np.mean(loss_ent_tmp), "\tloss reg:", np.mean(loss_reg_tmp))  
            print("val acc: ", np.mean(acc_tmp)*100, "%")
            print("----------------\n")
            loss_val_log.append([np.mean(loss_tmp), np.mean(loss_ent_tmp), np.mean(loss_reg_tmp)])
            acc_log.append(np.mean(acc_tmp))

            if np.mean(acc_tmp) > acc_best:
                model.save_weights(os.path.join(save_dir, "model_best.h5"))
                acc_best = np.mean(acc_tmp)
        
    np.save("log/loss_train_log.npy", np.array(loss_train_log))
    np.save("log/loss_val_log.npy", np.array(loss_val_log))
    np.save("log/acc_log.npy", np.array(acc_log))
    
    print("Best acc:", acc_best)
    print("done...")


# model
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
model = C3D()

# Create model directory
save_dir = "log/models"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load model
# y = model.predict(tf.convert_to_tensor(np.random.random((10, PARAMS['num_frames_per_clip'], 112, 112, 3)), dtype=tf.float32))
# model.load_weights(os.path.join(save_dir, "model_best.h5"))
# print("load pretrained weights")

# train
train(model, optimizer, save_dir=save_dir, training_step=5000)
