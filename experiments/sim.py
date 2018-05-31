import numpy as np
import math, os, imageio, pandas
import yaml, json, re
from scipy import ndimage, misc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.metrics import log_loss, mutual_info_score
from scipy.stats import spearmanr, entropy
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import data, io, filters, color
from PIL import Image

path = '/Users/romapatel/Documents/proto/'
path = '/nlp/data/romap/proto/'

# takes in 300d pretrained word embeddings
def word_similarity(m1, m2):
    metrics = {'cos_sim': 0, 'spearmanr': 0, 'abs_diff': 0, 'sq_diff': 0, 'mutual_info': 0, 'kl': 0}

    metrics['spearmanr'] = round(spearmanr(m1[0], m2[0])[0], 3)
    metrics['cos_sim'] = round(cos_sim(m1, m2)[0][0], 3)
    metrics['abs_diff'] = round(np.mean(m1-m2), 3)

    metrics['sq_diff'] = round(np.mean([math.pow((m1[0][i]-m2[0][i]), 2) for i in range(len(m1[0]))]), 3)    

    norm1, norm2 = m1, m2
    norm1, norm2 = [(1+i)/2 for i in norm1], [(1+i)/2 for i in norm2]
    norm1 = norm1 / np.sum(norm1); norm2 = norm2 / np.sum(norm2)
    #norm1 = [0.00001 + item[i] for item in norm1]
    #norm2 = [0.00001 + item[i] for item in norm1]
    
    #metrics['kl'] = round(min(10, entropy(norm1, norm2)), 3)
    metrics['mutual_info'] = round(mutual_info_score(norm1[0], norm2[0]), 3)
    
    return metrics

def get_embeddings(vocab, wvec_path):
    word_vecs = {}
    if wvec_path[-3:] == 'bin':
        with open(wvec_path, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    vec = np.fromstring(f.read(binary_len), dtype='float32')
                    word_vecs[word] = list(vec)  
                else:
                    f.read(binary_len)

    elif wvec_path[-3:] == 'txt':
        f = open(wvec_path, 'r'); lines = f.readlines()[1:]
        for line in lines:
            items = line.strip().split(' ')
            word, vec = items[0], [float(item) for item in items[1:]]
            if word in vocab: word_vecs[word] = vec
    return word_vecs


def evaluate():
    f = open(path + '/data/files/sem-vis-fin.txt', 'r')
    lines = f.readlines()

    vocab = [item for line in lines for item in line.split('\t')[0].split('#')]
    vocab = list(set(vocab))
    
    #wvecs = get_embeddings(vocab, wvec_paths['concept'])
    f = open(path + '/data/files/categories.txt', 'r')
    vocab = [line.strip() for line in f.readlines()]
    concept = get_embeddings(vocab, wvec_paths['concept'])
    print len(concept)

    google = get_embeddings(vocab, wvec_paths['google'])
    print len(google)

    glove = get_embeddings(vocab, wvec_paths['glove'])
    print len(glove)

    f = open('/nlp/data/romap/proto/data/sketchy/w2v/concept.json', 'w+')
    f.write(json.dumps(concept))
    f = open('/nlp/data/romap/proto/data/sketchy/w2v/google.json', 'w+')
    f.write(json.dumps(google))
    f = open('/nlp/data/romap/proto/data/sketchy/w2v/glove.json', 'w+')
    f.write(json.dumps(glove))
    return

    
    vocab = list(set(concept.keys()) & set(google.keys()) & set(glove.keys()))
    f_concept = open('/nlp/data/romap/proto/results/concept-sim.txt', 'w+')
    f_google = open('/nlp/data/romap/proto/results/google-sim.txt', 'w+')
    f_glove = open('/nlp/data/romap/proto/results/glove-sim.txt', 'w+')


    concept_vals, google_vals, glove_vals = [], [], []
    temp_lines = []
    for line in lines:
        print line
        word1, word2 = line.split('\t')[0].split('#')[0], line.split('\t')[0].split('#')[1]
        if len(set([word1, word2]) & set(vocab)) < 2: continue
        temp_lines.append(line)
        print 'concept'
        wvec1, wvec2 = np.reshape(concept[word1], (1, -1)), np.reshape(concept[word2], (1, -1))
        metrics1 = word_similarity(wvec1, wvec2)
        print metrics1; print '\n'
        temp = [line.split('\t')[0]]
        for key in sorted(metrics1):
            temp.append(metrics1[key])
        concept_vals.append(temp)
        print temp
        print 'google'
        wvec1, wvec2 = np.reshape(google[word1], (1, -1)), np.reshape(google[word2], (1, -1))
        metrics1 = word_similarity(wvec1, wvec2)
        print metrics1; print '\n'
        temp = [line.split('\t')[0]]
        for key in sorted(metrics1):
            temp.append(metrics1[key])
        google_vals.append(temp)

        print 'glove'
        wvec1, wvec2 = np.reshape(glove[word1], (1, -1)), np.reshape(glove[word2], (1, -1))
        metrics1 = word_similarity(wvec1, wvec2)
        temp = [line.split('\t')[0]]
        for key in sorted(metrics1):
            temp.append(metrics1[key])
        glove_vals.append(temp)
        
        print metrics1; print '\n\n'

    f_concept.write('pair\thuman-sim\thuman-vis\t'); f_google.write('pair\thuman-sim\thuman-vis\t'); f_glove.write('pair\thuman-sim\thuman-vis\t'); 
    for key in sorted(metrics1):
        f_concept.write(str(key) + '\t'); f_google.write(str(key) + '\t');  f_glove.write(str(key) + '\t')

    f_concept.write('\n')
    f_google.write('\n')
    f_glove.write('\n')
        
    f_concept.write('\n'); f_google.write('\n'); f_glove.write('\n');
    for row in range(len(concept_vals)):
        human_sim, human_vis = temp_lines[row].split('\t')[1], temp_lines[row].split('\t')[-1].strip()
        f_concept.write(concept_vals[row][0] + '\t' + human_sim + '\t' + human_vis + '\t')
        f_google.write(concept_vals[row][0] + '\t' + human_sim + '\t' + human_vis + '\t')
        f_glove.write(concept_vals[row][0] + '\t' + human_sim + '\t' + human_vis + '\t')

        for col in range(1, len(concept_vals[row])):
            f_concept.write(str(concept_vals[row][col]) + '\t')
            f_google.write(str((google_vals[row][col])) + '\t')
            f_glove.write(str((glove_vals[row][col])) + '\t')

        f_concept.write('\n')
        f_google.write('\n')
        f_glove.write('\n')

    glove = [[item for item in row[1:]] for row in glove_vals[1:]][1:]
    concept = [[item for item in row[1:]] for row in concept_vals[1:]][1:]
    google = [[item for item in row[1:]] for row in google_vals[1:]][1:]







    
def get_average(img_type):

    print 'img_type'; print img_type
    categories = [name for name in os.listdir(path + 'data/sketchy/categories/') if '.DS' not in name]
    average = {}
    count = 0
    for category in categories:
        f = open(path + 'data/sketchy/categories/' + category + '/' + img_type + '/train.txt', 'r')
        lines = f.readlines()[1:]
        cat_arr = np.zeros((256, 256, 3))
        for line in lines:
            img = Image.open(path + 'data/sketchy/figs/' + line.strip())
            #img = img.resize((64, 64))
            cat_arr += np.array(img).astype(np.float)
            count += 1

        cat_arr = cat_arr/(len(lines)-1)
        average[category] = cat_arr
        gray = color.rgb2gray(cat_arr); img = Image.fromarray(gray)
        img = img.convert('RGB')
        img.save(path + 'data/sketchy/average/' + img_type + '/' + category + '.jpg')
        

    return
    print 'count'; print count
    temp_lines = [];
    order = {'pair': [],
             'human-sim': [],
             'human-vis': [],
             'abs_diff': [],
             'sq_diff': []}
    f = open(path + '/data/files/sem-vis-fin.txt', 'r')
    lines = f.readlines()
    for line in lines:
        items = line.strip().split('\t')
        if len(items) < 3: continue
        order['pair'].append(items[0]); order['human-sim'].append(items[1]); order['human-vis'].append(items[2])
        word1, word2 = line.split('\t')[0].split('#')[0], line.split('\t')[0].split('#')[1]
        if len(set([word1, word2]) & set(average.keys())) < 2: continue
        temp_lines.append(line)
        metrics = image_similarity(average[word1], average[word2])
        order['abs_diff'].append(metrics['abs_diff'])
        order['sq_diff'].append(metrics['sq_diff'])
    # get average for a category, compute similarity

    print 'human-sim'
    print 'abs'
    print spearmanr(order['abs_diff'], order['human-sim'])[0]
    print 'sq'
    print spearmanr(order['sq_diff'], order['human-sim'])[0]
    print 'human-vis'
    print 'abs'
    print spearmanr(order['abs_diff'], order['human-vis'])[0]
    print 'sq'
    print spearmanr(order['sq_diff'], order['human-vis'])[0]
    print '\n\n'

    f = open('/nlp/data/romap/proto/results/img-metrics.json', 'w+')
    f.write(json.dumps(order))
    return average

def image_similarity(img1, img2):
    metrics = {'abs_diff': 0, 'sq_diff': 0}

    metrics['abs_diff'] = np.sum((img1.astype("float") - img2.astype("float"))) / float(img2.shape[0])
    metrics['sq_diff'] = np.sum((img1.astype("float") - img2.astype("float")) ** 2) / float(img2.shape[0] * img2.shape[1])

    return metrics

def get_matrix(img_type):
    categories = [name.split('.j')[0] for name in os.listdir(path + 'data/sketchy/average/sketch/') if '.DS' not in name]
    omit = ['tree', 'volcano', 'strawberry', 'pretzel', 'pizza', 'pineapple', 'hotdog', 'bread', 'banana', 'apple', 'hamburger', 'flower', 'geyser', 'pear', 'mushroom']
    categories = [item for item in categories if item not in omit]
    categories = sorted(set(categories))
    num = 10
    m = np.zeros((num, num))
    vals = {}; names = categories[90:100]

    names = ['tiger', 'lion', 'couch', 'table', 'swan', 'duck',
             'guitar', 'violin', 'lizard', 'crocodilian']

    names = ['motorcycle', 'bicycle', 'penguin', 'parrot', 'zebra', 'deer',
             'teapot', 'cup', 'pistol', 'rifle']
    
    for category in names:
        print category
        fpath = path + 'data/sketchy/average/' + img_type + '/' + category + '.jpg'
        img = Image.open(fpath)
        img = img.resize((64, 64))
        img = np.array(img).astype(np.float)
        img = color.rgb2gray(img) / 255
        a = [item for sublist in img for item in sublist]
        #print a
        vals[category] = a


    for i in range(num):
        for j in range(num):
            m[i][j] = spearmanr(vals[names[i]], vals[names[j]])[0]
 #           m[i][j] = cos_sim(np.reshape(vals[names[i]], (1, -1)), np.reshape(vals[names[j]], (1, -1)))[0]
 #           m[i][j] = round(np.mean(np.reshape(vals[names[i]], (1, -1)) - np.reshape(vals[names[j]], (1, -1))), 3)
    df = pandas.DataFrame(m)
    df.columns = names; df.index = names
    print df
    sns.set()
    ax = sns.heatmap(df, cmap="YlGnBu", annot=True)
    ax.set_ylabel(img_type)


 #   ax = sns.heatmap(df, cmap="YlGnBu", "Blues")
    plt.show()
    
if __name__ == '__main__':
    global wvec_paths
    wvec_paths = {'google': '/nlp/data/corpora/GoogleNews-vectors-negative300.bin',
                                   'legal': '/nlp/data/romap/ambig/w2v/w2v100-300.txt',
                                   'concept': '/nlp/data/romap/conceptnet/numberbatch-en-17.06.txt',
                                   'glove': '/nlp/data/romap/w2v/glove.840B.300d.txt'
                                   }

 #   get_matrix('sketch')
#    get_matrix('photo')
 #   get_average('sketch')
 #   get_average('photo')

    evaluate()

    ''' 
    wvec1, wvec2 = np.reshape([0.5, 0.2, 0.3], (1, -1)), np.reshape([0.4, 0.1, 0.5], (1, -1))
    metrics = word_similarity(wvec1, wvec2)
    print metrics

    wvec1, wvec2 = np.reshape([0.5, 0.2, 0.3], (1, -1)), np.reshape([0.5, 0.2, 0.3], (1, -1))
    metrics = word_similarity(wvec1, wvec2)
    print metrics
    '''
