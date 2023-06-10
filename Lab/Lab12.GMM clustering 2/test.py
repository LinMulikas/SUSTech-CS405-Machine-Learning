# Dependency
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import tqdm

from PIL import Image

COLORS = [
    (255, 0, 0),   # red
    (0, 255, 0),  # green
    (0, 0, 255),   # blue
    (255, 255, 0), # yellow
    (255, 0, 255), # magenta
]

import cv2

def loadImage(image_path):
    image = np.array(cv2.imread(image_path))
    h, w, c = image.shape
    image = image.reshape((h*w, c))

    _mean = np.mean(image, axis = 0)
    _std = np.std(image, axis = 0)
    # TODO: please normalize image_pixl using Z-score
    normed_img = (image - _mean)/_std
        
    return h, w, c, normed_img

class GMM:
    def __init__(self, ncomp, initial_mus, initial_covs, initial_priors):
        """
        :param ncomp:           the number of clusters
        :param initial_mus:     initial means
        :param initial_covs:    initial covariance matrices
        :param initial_priors:  initial mixing coefficients
        """
        self.ncomp = ncomp
        self.mus = np.asarray(initial_mus)
        self.covs = np.asarray(initial_covs)
        self.priors = np.asarray(initial_priors)

    def inference(self, datas:np.ndarray):
        """
        E-step
        :param datas:   original data
        :return:        posterior probability (gamma) and log likelihood
        """
        probs = []
        self.mus = datas[np.random.choice(datas.shape[0], 5)]
        self.covs = np.full(fill_value = 0.1, shape = (5, 3, 3))
        self.priors = np.repeat(1/5, 5)
        for i in range(self.ncomp):
            mu, cov, prior = self.mus[i, :], self.covs[i, :, :], self.priors[i]
            # 出现在 cls = i 的概率
            prob = prior * multivariate_normal.pdf(
                datas, mean=mu, cov=cov, allow_singular=True
            )
            probs.append(np.expand_dims(prob, -1))
        
        # 生成一个 N * n_cls 的矩阵
        preds = np.concatenate(probs, axis=1)

        # TODO: calc log likelihood
        log_likelihood = np.log(preds)

        # TODO: calc gamma
        gamma = np.ndarray((datas.shape[0], 5))
        summ = np.sum(preds, axis=1)
        for i in range(0, datas.shape[0]):
            for j in range(0, 5):
                gamma[i, j] = (self.priors[j] * preds[i, j])/summ[i]


        return gamma, log_likelihood

    def update(self, datas, gamma):
        """
        M-step
        :param datas:   original data
        :param gamma:    gamma
        :return:
        """
        new_mus, new_covs, new_priors = [], [], []
        soft_counts = np.sum(gamma, axis=0)
        for i in range(self.ncomp):
            # TODO: calc mu
            new_mu = None
            new_mus.append(new_mu)

            # TODO: calc cov
            new_cov = None
            new_covs.append(new_cov)

            # TODO: calc mixing coefficients
            new_prior = None
            new_priors.append(new_prior)

        self.mus = np.asarray(new_mus)
        self.covs = np.asarray(new_covs)
        self.priors = np.asarray(new_priors)

    def fit(self, data, iteration):
        prev_log_liklihood = None

        bar = tqdm.tqdm(total=iteration)
        for i in range(iteration):
            gamma, log_likelihood = self.inference(data)
            self.update(data, gamma)
            if prev_log_liklihood is not None and abs(log_likelihood - prev_log_liklihood) < 1e-10:
                break
            prev_log_likelihood = log_likelihood

            bar.update()
            bar.set_postfix({"log likelihood": log_likelihood})         
            
            
from PIL import Image
import matplotlib.pyplot as plt


def visualize(gmm, image, ncomp, ih, iw):
    beliefs, log_likelihood = gmm.inference(image)
    map_beliefs = np.reshape(beliefs, (ih, iw, ncomp))
    segmented_map = np.zeros((ih, iw, 3))
    for i in range(ih):
        for j in range(iw):
            hard_belief = np.argmax(map_beliefs[i, j, :])
            segmented_map[i, j, :] = np.asarray(COLORS[hard_belief]) / 255.0
            
            
    plt.imshow(segmented_map)
    plt.show()
    
    
    
h, w, c, image = loadImage("/Users/mulikas/Codes/CS405 ML/Lab/Lab12.GMM clustering 2/data/original/sample.png")
n = image.shape[0]

indices = np.random.choice(range(0, n), 5, replace=False)
X = image[indices]

gmm = GMM(5, 1, 0.1, 1/5)
visualize(gmm, image, 5, h, w)