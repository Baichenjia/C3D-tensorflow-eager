import numpy as np
import matplotlib.pyplot as plt

def smooth(x):
    lenth = x.shape[0]
    res = [x[0]]
    for i in range(lenth-1):
        res.append(res[-1] * 0.9 + 0.1 * x[i+1])
    return res

def plot_res():
    loss_train_np = np.load("log/loss_train_log.npy")
    loss_val_np = np.load("log/loss_val_log.npy")
    acc_log_np = np.load("log/acc_log.npy")
    plt.figure(figsize=(8,8)) 
    plt.plot(np.arange(loss_train_np.shape[0]), smooth(loss_train_np[:, 0]))
    plt.plot(np.arange(loss_train_np.shape[0]), smooth(loss_train_np[:, 1]))
    plt.plot(np.arange(loss_train_np.shape[0]), smooth(loss_train_np[:, 2]))
    plt.legend(["train loss", "loss entropy", "loss reg"])
    plt.grid()
    plt.savefig("log/pic/loss_train.jpg")
    plt.close()

    plt.figure(figsize=(8,8)) 
    plt.plot(np.arange(loss_val_np.shape[0]), smooth(loss_val_np[:, 0]))
    plt.plot(np.arange(loss_val_np.shape[0]), smooth(loss_val_np[:, 1]))
    plt.plot(np.arange(loss_val_np.shape[0]), smooth(loss_val_np[:, 2]))
    plt.legend(["val loss", "loss entropy", "loss reg"])
    plt.grid()
    plt.savefig("log/pic/loss_val.jpg")
    plt.close()

    plt.figure(figsize=(8,8)) 
    print(acc_log_np.shape)
    plt.plot(np.arange(acc_log_np.shape[0]), smooth(acc_log_np))
    plt.legend(["val acc"])
    plt.grid()
    plt.savefig("log/pic/acc_val.jpg")
    plt.close()

# plot
if __name__ == '__main__':
    plot_res()
