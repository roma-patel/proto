import numpy as np
import math, os, imageio, pandas
from collections import defaultdict
from scipy import ndimage, misc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import log_loss
import yaml, json, re
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from image import Image

path = '/Users/romapatel/Documents/proto/'
#path = '/nlp/data/romap/proto/'

img_size = 256;
num = 3;

def get_images(category_path, img_type, gray):
    print category_path
    
    f = open(category_path + '/' + img_type + '/filenames.txt')
    fnames = [line.strip() for line in f.readlines()[1:]]
    avg = np.zeros((256, 256, 3))
    for fname in fnames[:num]:
        fpath = path + '/data/sketchy/efigs/' + fname
        if os.path.isfile(fpath) is False: continue
        img = Image(fpath, True)
        img = img.get_pixels()
        print img
        avg += np.array(img).astype(np.uint8)
    avg = np.array(avg / (1.0*num)).astype(np.uint8)
    return avg

def random():
    path = os.getcwd(); cat_path = path + '/data/sketchy/ecategories/'
    # path to random experimental 10 images for each category
    categories = os.listdir(cat_path)
    for category in categories[:3]:
        get_images(cat_path + category, 'sketch', True)
    
    '''
    for category in sorted(categories)[:2]:
        if 'pixel' in category or '.DS' in category: continue
        files = os.listdir(path + category)
        for filename in files[:2]:
            if '.DS' in filename: continue
            a = Image(path + category + '/' + filename)
            for row in a.get_pixels():
                print row
            for row in a.get_pixel_features():
                print row
            plt.imshow(a.get_pixels(), cmap='gray')
            plt.show()
    '''
    '''
    df = pandas.DataFrame(sp, index=res['indices'], columns=res['indices'] )
    sns.set()
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu")
    ax.set_xlabel('Spearman Correlation')
    
    plt.show()
    '''


    
if __name__ == '__main__':
    random()
