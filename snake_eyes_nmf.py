import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import time


def read_vectors(*filenames):
    data = np.vstack(
        tuple(np.fromfile(filename, dtype=np.uint8).reshape(-1, 401)
              for filename in filenames))
    return data[:, 1:], data[:, 0] - 1

def gen_image(arr):
    two_d = (np.reshape(arr, (20, 20))).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest', cmap='gray')
    return plt


def show_images(array, max_images=20, title="Figure"):
    # sub-image width and height
    w = 20
    h = 20

    max = array.shape[1]
    if max_images < array.shape[1]:
        max = max_images

    # how many sub-images
    columns = 4
    if max % 4 == 0:
        rows = int(max / 4)
    else:
        rows = int(max / 4) + 1

    fig = plt.figure(figsize=(columns * 2, rows * 2))
    fig.suptitle(title)

    for i in range(1, max + 1):
        img = array[i - 1, :].reshape(20, 20)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    return plt

td1 = time.time()
X_train, y_train = read_vectors(*["./snake-eyes-data/snakeeyes_{:02d}.dat".format(nn) for nn in range(10)])
X_test, y_test = read_vectors("./snake-eyes-data/snakeeyes_test.dat")
td2 = time.time()
print(X_train.shape)
print("Loading the data took {:.2f}s".format(td2 - td1))

show_images(X_train,20).show()