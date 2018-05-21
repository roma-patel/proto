import numpy as np
import math, os, imageio
from collections import defaultdict
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import yaml, json, re
from sklearn.decomposition import PCA

size = 1111;

class Image:
    def __init__(self, path):
        self.path = path
        self.pixels = []
        self.pixel_features = []
        self.label = []
        self.label_features = []

    def get_pixels(self):
        self.pixels = ndimage.imread(self.path)
        return self.pixels

    def get_pixel_features(self):
        self.pixel_features = np.array([[0 if item == 255 else 1 for item in row] for row in self.pixels])
        return self.pixel_features

    def get_label(self):
        self.label = self.path.split('/')[-2]
        return self.label

    def get_label_features(self):
        # wvec path
        #wvecs = get_wvecs()
        self.label_features = wvecs[self.label]
        return self.label_features


        
if __name__ == '__main__':
    path = '/Users/romapatel/Desktop/sketches_png/'
    categories = os.listdir(path)
    for category in sorted(categories)[:2]:
        if 'pixel' in category or '.DS' in category: continue
        files = os.listdir(path + category)
        for filename in files[:2]:
            if '.DS' in filename: continue
            a = Image(path + category + '/' + filename)
            plt.imshow(a.get_pixels(), cmap='gray')
            plt.show()

            plt.imshow(a.get_pixel_features())
            plt.show()
            plt.savefig('/Users/romapatel/Desktop/a.png')
