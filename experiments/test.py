import numpy as np
import math, os, imageio
from collections import defaultdict
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import log_loss
import yaml, json, re
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from image import Image
from PIL import Image as im

path = '/Users/romapatel/Documents/proto/'
path = '/nlp/data/romap/proto/'

img_size = 1111;

def get_categories():
    f = open(path + 'data/mcrae-sketch.txt', 'r')
    categories = [line.strip() for line in f.readlines()]

    f = open(path + 'data/mcrae.json', 'r')
    for line in f:
        cat_dict = json.loads(line)

    fine, coarse = [], []
    coarse = cat_dict.keys(); fine = [cat for cat in categories if cat not in coarse]

    '''
    for cat in fine:
        flag = True
        for key in cat_dict:
            if cat in cat_dict[key].keys():
                flag = False; print cat; print key; print '\n'
        if flag is True: print cat
    '''

    print sorted(cat_dict.keys())
    '''
    train, test = {}, {}
    f = 0
    for cat in cat_dict:
        print cat
        items = cat_dict[cat].keys(); items = [item for item in items if cat_dict[cat][item] > 2 and item in fine]
        print sorted(set(cat_dict[cat])&set(items)); print len(set(cat_dict[cat])&set(items)); print '\n\n'
        f += len(set(cat_dict[cat])&set(items))

    print f
    '''
    
    return coarse, fine, cat_dict

def create_splits():
    coarse, fine, cat_dict = get_categories()
    # one: all categories included in train, within each category split 90:10
    split = 9*8
    train, test = {}, {}
    for category in fine:
        cat_path = path + 'data/tu-berlin/sketches_png/' + category + '/'
        if os.path.isdir(cat_path) is False: continue
        print category
        filenames = [item for item in os.listdir(cat_path) if 'png' in item]
        train[category] = filenames[:split]
        test[category] = filenames[split:]


    f = open(path + 'data/tu-berlin/train_1.json', 'w+')
    f.write(json.dumps(train))

    f = open(path + 'data/tu-berlin/test_1.json', 'w+')
    f.write(json.dumps(test))
    
            
def reduce_dim(m):
    return m

def similarity(m1, m2):
    return sim

def create_prototypes():
    coarse, fine, cat_dict = get_categories()
    f = open(path + 'data/tu-berlin/train_1.json', 'r')
    for line in f: train = json.loads(line)

    f = open(path + 'data/tu-berlin/test_1.json', 'r')
    for line in f: test = json.loads(line)

    ###### change this!
    
    all_cats = sorted(train.keys())
    prbins, prraws = {}, {}
    for category in all_cats:
        cat_path = path + 'data/tu-berlin/sketches_png/' + category + '/'
        if os.path.isdir(cat_path) is False: continue
        filenames = train[category]
        prbin, prraw = np.zeros((img_size, img_size)), np.zeros((img_size, img_size))
        
        for filename in filenames:
            if '.DS' in filename: continue
            a = Image(cat_path + filename);
            prbin += a.get_pixel_features()
            prraw += a.get_pixels()
            
        prbin /= len(filenames)
        prraw /= len(filenames)

        prbins[category] = [list(item) for item in prbin]
        prraws[category] = [list(item) for item in prraw]

    
    f = open(path + 'data/tu-berlin/prototypes_bin_2.json', 'w+')
    f.write(json.dumps(prbins))

    f = open(path + 'data/tu-berlin/prototypes_raw_2.json', 'w+')
    f.write(json.dumps(prraws))
    
def prototype_model(pixel_type):
    coarse, fine, cat_dict = get_categories()
    f = open(path + 'data/tu-berlin/train_1.json', 'r')
    for line in f: train = json.loads(line)

    f = open(path + 'data/tu-berlin/test_1.json', 'r')
    for line in f: test = json.loads(line)

    ###### change this!
    all_cats = sorted(train.keys())


    all_cats = sorted(train.keys())[:10]

    # change this
    #f = open(path + 'data/tu-berlin/prototypes_' + pixel_type + '.json', 'r')
    f = open('/Users/romapatel/Desktop/prototypes_bin_2.json', 'r')
    for line in f: prototypes = json.loads(line)


    results = {}; num = len(all_cats)
    cos_matrix, sp_matrix = np.zeros((num, num)), np.zeros((num, num))
    abs_matrix, ce_matrix = np.zeros((num, num)), np.zeros((num, num))

    for i in range(len(all_cats)):
        category = sorted(test.keys())[i]
        print category
        
        cat_path = path + 'data/tu-berlin/sketches_png/' + category + '/'
        if os.path.isdir(cat_path) is False: continue
        filenames = test[category]

        for filename in filenames:
            if '.DS' in filename: continue
            print filename
            a = Image(cat_path + filename);
            if pixel_type == 'bin':
                pixels = a.get_pixel_features()

            else:
                pixels = a.get_pixels()
            flat_pixels = [val for sublist in pixels for val in sublist]


            cos_temp, sp_temp = [], []
            for j in range(len(all_cats)):
                cat = sorted(test.keys())[j]
                prototype = prototypes[cat]
                flat_prototype = [val for sublist in prototype for val in sublist]

                cos_matrix[i][j] += np.mean(cos_sim(pixels, prototype))
                sp_matrix[i][j] += spearmanr(flat_pixels, flat_prototype)[0]
                abs_matrix[i][j] += np.mean(pixels-prototype)
                ce_matrix[i][j] += np.mean(log_loss(pixels, prototype))


    cos_matrix = [list(item) for item in cos_matrix]
    sp_matrix = [list(item) for item in sp_matrix]
    abs_matrix = [list(item) for item in abs_matrix]
    ce_matrix = [list(item) for item in ce_matrix]


    f = open(path + 'results/tu-berlin/prototype/' + pixel_type + '2.json', 'w+')

    results = {'cos_sim': list(cos_matrix), 'spearman': list(sp_matrix), 'abs': list(abs_matrix), 'ce': list(ce_matrix), 'indices': all_cats}
    f.write(json.dumps(results))


def exemplar_model(pixel_type):
    coarse, fine, cat_dict = get_categories()
    f = open(path + 'data/tu-berlin/train_1.json', 'r')
    for line in f: train = json.loads(line)

    f = open(path + 'data/tu-berlin/test_1.json', 'r')
    for line in f: test = json.loads(line)

    all_cats = sorted(train.keys())
    f = open(path + 'data/tu-berlin/prototypes_' + pixel_type + '_80.json', 'r')
    for line in f: prototypes = json.loads(line)

    for x_cat in test.keys():
        for y_cat in train.keys():
            cos_matrix, sp_matrix = np.zeros((num, num)), np.zeros((num, num))
            abs_matrix, ce_matrix = np.zeros((num, num)), np.zeros((num, num))

            a_x = Image(cat_path + filename);
            a_y = Image(cat_path + filename);


if __name__ == '__main__':
    #create_splits()
    create_prototypes()
    prototype_model('bin')
    #prototype_model('raw')

    #get_categories()
    #run()
    
    '''
    cat_path = path + 'data/tu-berlin/sketches_png/'
    categories = os.listdir(cat_path)
    for category in sorted(categories)[:2]:
        if 'pixel' in category or '.DS' in category: continue
        files = os.listdir(cat_path + category)
        for filename in files[:2]:
            if '.DS' in filename: continue
            a = Image(cat_path + category + '/' + filename)
            plt.imshow(a.get_pixels(), cmap='gray')
            plt.show()

            #plt.imshow(a.get_pixel_features())
            #plt.show()
    '''
