import numpy as np
import math, os, imageio
'''
Segregates, resizes (if needed) and splits into train and test
Also creates stats files for each category (sketchs vs. photo images)
'''
from collections import defaultdict
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import yaml, json, re
from sklearn.decomposition import PCA
from PIL import Image
import os

def create_files():
    # sketchy dataset figure folder path
    path = '/nlp/data/romap/proto/data/'
    names = os.listdir(path + 'sketchy/figs/')

    sketches = [name for name in names if name.endswith('png')]
    photos = [name for name in names if name.endswith('jpg')]

    categories = {}
    f = open(path + 'stats/info/stats.csv', 'r')
    for line in f.readlines()[1:]:
        items = line.split(','); cat, imgnet = items[1], items[2].split('_')[0]
        if cat not in categories.keys(): categories[cat] = []
        categories[cat].append(imgnet)

    f = open(path + 'stats/info/cat_imgnet.json', 'w+')
    f.write(json.dumps(categories)); f.close()

    for category in categories:
        old_category = category
        category = re.sub(' ', '_', category.strip())
        if os.path.isdir(path + 'sketchy/categories/' + category) is False:
            os.mkdir(path + 'sketchy/categories/' + category + '/')
            os.mkdir(path + 'sketchy/categories/' + category + '/sketch/')
            os.mkdir(path + 'sketchy/categories/' + category + '/photo/')

        sk = [sketch for sketch in sketches if sketch.split('_')[0] in categories[old_category]]
        ph = [photo for photo in photos if photo.split('_')[0] in categories[old_category]]

        #print sk; print ph;
        #print '\n\n\n'
            
        f = open(path + 'sketchy/categories/' + category + '/sketch/filenames.txt', 'w+')
        f.write(str(len(sk)) + '\n')
        for sketch in sk:
            f.write(sketch + '\n')
        f = open(path + 'sketchy/categories/' + category + '/photo/filenames.txt', 'w+')
        f.write(str(len(ph)) + '\n')
        for photo in ph:
            f.write(photo + '\n')

    # train test split
    for category in categories:
        print category
        category = re.sub(' ', '_', category.strip())

        f = open(path + 'sketchy/categories/' + category + '/sketch/filenames.txt', 'r')
        lines = [line.strip() for line in f.readlines()[1:]]
        test = lines[:len(lines)/10]
        train = lines[len(lines)/10:]
        f = open(path + 'sketchy/categories/' + category + '/sketch/train.txt', 'w+')
        f.write(str(len(train)) + '\n')
        for name in train:
            f.write(name + '\n')

        f = open(path + 'sketchy/categories/' + category + '/sketch/test.txt', 'w+')
        f.write(str(len(test)) + '\n')
        for name in test:
            f.write(name + '\n')
            
        f = open(path + 'sketchy/categories/' + category + '/photo/filenames.txt', 'r')
        lines = [line.strip() for line in f.readlines()[1:]]
        test = lines[:len(lines)/10]
        train = lines[len(lines)/10:]
        f = open(path + 'sketchy/categories/' + category + '/photo/train.txt', 'w+')
        f.write(str(len(train)) + '\n')
        for name in train:
            f.write(name + '\n')

        f = open(path + 'sketchy/categories/' + category + '/photo/test.txt', 'w+')
        f.write(str(len(test)) + '\n')
        for name in test:
            f.write(name + '\n')
            
def create_train_test():
    for category in categories:
        f = open(path + 'sketchy/categories/' + category + '/sketch/filenames.txt', 'r')
        lines = f.readlines()[1:]
        test = lines[:len(lines)/10]
        train = lines[len(lines)/10:]
        f = open(path + 'sketchy/categories/' + category + '/sketch/train.txt', 'w+')
        f.write(str(len(train)) + '\n')
        for name in train:
            f.write(name + '\n')

        f = open(path + 'sketchy/categories/' + category + '/sketch/test.txt', 'w+')
        f.write(str(len(test)) + '\n')
        for name in test:
            f.write(name + '\n')
            
        f = open(path + 'sketchy/categories/' + category + '/photo/filenames.txt', 'r')
        lines = f.readlines()[1:]
        test = lines[:len(lines)/10]
        train = lines[len(lines)/10:]
        f = open(path + 'sketchy/categories/' + category + '/photo/train.txt', 'w+')
        f.write(str(len(train)) + '\n')
        for name in train:
            f.write(name + '\n')

        f = open(path + 'sketchy/categories/' + category + '/photo/test.txt', 'w+')
        f.write(str(len(test)) + '\n')
        for name in test:
            f.write(name + '\n')

def create_temp():
    fname = {}; path = '/nlp/data/romap/proto/data/sketchy/categories/'
    categories = os.listdir(path); names = []; 
    for category in categories:
        photo = path + category + '/photo/train.txt';
        f = open(photo, 'r'); photos = f.readlines()[1:]
        np.random.shuffle(photos)
        sketch = path + category + '/sketch/train.txt';
        f = open(sketch, 'r'); sketches = f.readlines()[1:]
        np.random.shuffle(sketches)
        
        fname[category] = {'photo': [line.strip() for line in photos[:8]], 'sketch': [line.strip() for line in sketches[:8]]}
        test = {'photo': [line.strip() for line in photos[8:10]], 'sketch': [line.strip() for line in sketches[8:10]]} 
        if os.path.isdir('/nlp/data/romap/proto/data/sketchy/ecategories/' + category) is False:
            os.mkdir('/nlp/data/romap/proto/data/sketchy/ecategories/' + category)
            os.mkdir('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/photo/')
            os.mkdir('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/sketch/')

        ftrain = open('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/photo/train.txt', 'w+')
        ftest = open('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/photo/test.txt', 'w+')
        fall = open('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/photo/filenames.txt', 'w+')
        ftrain.write('8\n'); ftest.write('2\n'); fall.write('10\n')
        for name in fname[category]['photo']:
            ftrain.write(name + '\n'); fall.write(name + '\n')
            names.append(name)
        for name in test['photo']:
            ftest.write(name + '\n'); fall.write(name + '\n')
            names.append(name)

        ftrain = open('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/sketch/train.txt', 'w+')
        ftest = open('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/sketch/test.txt', 'w+')
        fall = open('/nlp/data/romap/proto/data/sketchy/ecategories/' + category + '/sketch/filenames.txt', 'w+')
        ftrain.write('8\n'); ftest.write('2\n'); fall.write('10\n')
        for name in fname[category]['sketch']:
            ftrain.write(name + '\n'); fall.write(name + '\n')
            names.append(name)
        for name in test['sketch']:
            ftest.write(name + '\n'); fall.write(name + '\n')
            names.append(name)
            
    f = open('/nlp/data/romap/proto/data/sketchy/efnames.txt', 'w+')
    for name in names: f.write(name + '\n')

                
if __name__ == '__main__':
    create_temp()
    ##create_files()
    ##create_train_test()
    
