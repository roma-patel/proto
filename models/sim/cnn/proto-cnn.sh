source /home1/r/romap/crf/crf_task/bin/activate
#python /nlp/data/romap/proto/models/cnn/data_loader.py
#python /nlp/data/romap/proto/models/cnn/main.py -epochs 90 -b 256 -lr 0.1 -momentum 0.9 -p 10 
cd /nlp/data/romap/proto/models/sim/cnn/
python cifar-test.py
