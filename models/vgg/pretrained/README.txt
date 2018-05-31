
FILES:

map-vis_avg_vgg128_glove_lin.csv and map-vis_avg_vgg128_glove_nn.csv are the linear and neuralnet mapped embeddings.
MODEL_lin.pkl and MODEL_nn.pkl are the linear and neuralnet models stored in pkl format.



DEPENDENCIES: 

scikit-learn neural network Python library (the models are learned with this library) 



GENERATE:

The embeddings already cover all the vocabulary present in the test sets of our experiments. However, one can generate the multimodal (mapped) representation of any word with the models provided. It is necessary however that the input text vector is a GloVe vector: http://nlp.stanford.edu/projects/glove/ from the largest corpus (300-d). 



For example, the Python script to generate the multimodal embedding for one word (NewWord) is:
 


import numpy as np

import pickle

model = pickle.load(open('my_folder/MODEL_lin.pkl', 'rb')) #call the model, stored in my_folder


NewWord = np.array([300*[1]]) # a toy example (replace this vector by an actual GloVe vector)

imagined = model.predict(NewWord) # generate "imagined" embedding for the new word (by predicting)


imagined = imagined / np.linalg.norm(imagined) # L2-normalize the imagined vector

NewWord = NewWord / np.linalg.norm(NewWord) # L2-normalize the GloVe vector

MAP_C = np.concatenate( (NewWord, imagined) ,axis=1) # concatenate text vector to the imagined vector to obtain the MAP_C representation 
