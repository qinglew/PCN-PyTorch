import os
import random


DATASET_PATH = '/media/rico/BACKUP/Dataset/ShapeNetForPCN'
categories = [cat for cat in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, cat))]
train_file = open('dataset/train.list', 'w')
val_file = open('dataset/val.list', 'w')
test_file = open('dataset/test.list', 'w')
for cat in categories:
    cat_dir = os.path.join(DATASET_PATH, cat)
    models = os.listdir(cat_dir)
    random.shuffle(models)
    train, val, test = models[:500], models[500:600], models[600:700]
    for filename in train:
        train_file.write(cat + '/' + filename + '\n')
    for filename in val:
        val_file.write(cat + '/' + filename + '\n')
    for filename in test:
        test_file.write(cat + '/' + filename + '\n')
train_file.close()
val_file.close()
test_file.close()
