import numpy as np
import math, os, imageio
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
import math
import pandas

path = '/Users/romapatel/Documents/proto/'
#path = '/nlp/data/romap/proto/'

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

def evaluate(pixel_type, num):
    
    print 'Inside evaluate\n'
    #f = open(path + 'results/tu-berlin/prototype/bin.json', 'r')
    f = open(path + 'results/tu-berlin/prototype/' + pixel_type + '_' + str(num) +'.json', 'r')

    for line in f:
        res = json.loads(line)

    sp = res['spearman']
    ab = res['abs']
    cs = res['cos_sim']
    ce = res['ce']

    for i in range(len(ab)):
        ab[i] = [math.fabs(val) for val in ab[i]]
    for cat in res['indices']:
        print cat

    print len(sp)
    df = pandas.DataFrame(sp, index=res['indices'], columns=res['indices'] )
    sns.set()
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu")
    ax.set_xlabel('Spearman Correlation')
    
    plt.show()

    df = pandas.DataFrame(ab, index=res['indices'], columns=res['indices'] )
    sns.set()
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu")
    ax.set_xlabel('Absolute Difference')
    
    plt.show()

    df = pandas.DataFrame(cs, index=res['indices'], columns=res['indices'] )
    sns.set()
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu")
    ax.set_xlabel('Cosine Similarity')
    
    plt.show()

    df = pandas.DataFrame(ce, index=res['indices'], columns=res['indices'] )
    sns.set()
    ax = sns.heatmap(df, annot=True, cmap="YlGnBu")
    ax.set_xlabel('Cross Entropy')
    
    plt.show()

def prototype_model(pixel_type, num):
    coarse, fine, cat_dict = get_categories()
    f = open(path + 'data/tu-berlin/train_1.json', 'r')
    for line in f: train = json.loads(line)

    f = open(path + 'data/tu-berlin/test_1.json', 'r')
    for line in f: test = json.loads(line)
    
    print 'Inside prototype model\n'
    prototypes = {}
    f = open('/Users/romapatel/Desktop/prototypes_20.json', 'r')
    for line in f.readlines()[:num]:
        temp = json.loads(line)
        prototypes[temp['category']] = temp

    all_cats = sorted(prototypes.keys())

    results = {}; num = len(all_cats)
    cos_matrix, sp_matrix = np.zeros((num, num)), np.zeros((num, num))
    abs_matrix, ce_matrix = np.zeros((num, num)), np.zeros((num, num))

    for i in range(len(all_cats)):
        category = sorted(prototypes.keys())[i]
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
                cat = sorted(prototypes.keys())[j]
                prototype = prototypes[cat]['prototype_arr']
                print len(prototype)
                flat_prototype = [val for sublist in prototype for val in sublist]

                cos_matrix[i][j] += np.mean(cos_sim(pixels, prototype))
                sp_matrix[i][j] += spearmanr(flat_pixels, flat_prototype)[0]
                abs_matrix[i][j] += np.mean(pixels-prototype)
                ce_matrix[i][j] += np.mean(log_loss(pixels, prototype))









            break

    cos_matrix = [list(item) for item in cos_matrix]
    sp_matrix = [list(item) for item in sp_matrix]
    abs_matrix = [list(item) for item in abs_matrix]
    ce_matrix = [list(item) for item in ce_matrix]


    f = open(path + 'results/tu-berlin/prototype/' + pixel_type + '_' + str(num) +'.json', 'w+')

    results = {'cos_sim': list(cos_matrix), 'spearman': list(sp_matrix), 'abs': list(abs_matrix), 'ce': list(ce_matrix), 'indices': all_cats}
    f.write(json.dumps(results))
    
if __name__ == '__main__':
    '''
    num = 10
    prototype_model('bin', num)
    num = 20
    prototype_model('bin', num)
    num = 30
    prototype_model('bin', num)
    num = 40
    prototype_model('bin', num)
    num = 50
    prototype_model('bin', num)
    '''

    num = 20
    #prototype_model('bin', num)
    evaluate('bin', num)
