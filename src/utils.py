import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


def load_data(img_path, label_path, gray=True):
    ''' Loads images and labels from the path '''
    # get images
    img_list = []
    img_names = []
    for img_name in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, img_name))
        img_list.append(img)
        img_names.append(img_name)

    if gray:
        img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]

    # create label dict
    label_dict = {}
    with open(label_path, 'r') as f:
        for line in f:
            name = line.strip().split(' ')[0][:-4] + '_aligned.jpg'
            label = int(line.strip().split(' ')[1])
            label_dict[name] = label

    # get labels from dict
    label_list = [label_dict[name] for name in img_names]

    return img_list, label_list


def display_n_images(img_list, label_list, n):
    ''' Displays n images with their labels '''
    rows = int(n**0.5)
    cols = int(n/rows)
    n = rows*cols

    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
    ax = axs.ravel()

    indices = np.random.choice(len(img_list), n, replace=False)
    for i, idx in enumerate(indices):
        ax[i].imshow(img_list[idx], cmap='gray')
        ax[i].set_title(label_list[idx])
        ax[i].axis('off')
    fig.tight_layout()
    plt.show()


def create_SIFT_features(img_list, label_list, print_example=False):  # code from labs
    ''' Creates SIFT features '''
    print('Creating SIFT features...')
    des_list = []
    des_label_list = []
    sift = cv2.SIFT_create()

    if print_example:
        fig, ax = plt.subplots(1, 4, figsize=(8, 6), sharey=True)

    for i, img in enumerate(img_list):
        kp, des = sift.detectAndCompute(img, None)

        # Show results for first 4 images
        if i < 4 and print_example:
            img_with_SIFT = cv2.drawKeypoints(img, kp, img)
            ax[i].imshow(img_with_SIFT)
            ax[i].set_axis_off()

        if des is not None:
            des_list.append(des)
            des_label_list.append(label_list[i])

    return des_list, des_label_list


def cluster_descriptors(des_list, k):
    ''' Clusters descriptors '''
    print('Clustering descriptors...')
    des_array = np.vstack(des_list)
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    kmeans = kmeans.fit(des_array)
    return kmeans


def convert_des_to_hist(des_list, kmeans, k, plot=True):
    ''' Converts descriptors to histograms '''
    print('Converting descriptors to histograms...')
    hist_list = []
    idx_list = []
    for des in des_list:
        hist = np.zeros(k)
        idx = kmeans.predict(des)
        idx_list.append(idx)

        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)
        idx_list.append(idx)

    hist_array = np.vstack(hist_list)

    # plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(np.array(idx_list, dtype=object), bins=k)
        ax.set_title('Codewords occurrence in training set')
        plt.show()
    return hist_array


def convert_test_to_hist(X_test_list, kmeans, k):

    hist_list = []
    sift = cv2.SIFT_create()
    for i in range(len(X_test_list)):
        kp, des = sift.detectAndCompute(X_test_list[i], None)

        if des is not None:
            hist = np.zeros(k)
            idx = kmeans.predict(des)
            for j in idx:
                hist[j] = hist[j] + (1 / len(des))

            # hist = scale.transform(hist.reshape(1, -1))
            hist_list.append(hist)

        else:
            hist_list.append(None)

    # Remove potential cases of images with no descriptors
    idx_not_empty = [i for i, x in enumerate(hist_list) if x is not None]
    hist_list = [hist_list[i] for i in idx_not_empty]
    y_test = [y_test[i] for i in idx_not_empty]
    hist_array = np.vstack(hist_list)
    return hist_array
