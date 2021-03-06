import gzip
import os
from PIL import Image, ImageOps
import cPickle
import numpy as np
import pdb
from pylearn2.utils import serial

base = './MIT'
excetionList = ["bakery","bar","buffet","casino","deli","dining_room","fastfood_restaurant",
                "grocerystore","pantry","restaurant","restaurant_kitchen","winecellar"]
folders = [entry for entry in os.listdir(base)]
target = './Food'

TEST_SIZE = 1166
TRAIN_SIZE = 7000

dataTestX = []
dataTrainX = []

counter = 0

def save_zipped_pickle(filename, obj, protocol=2):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)

for folder in folders:
    if folder not in excetionList:

        path = base+"/"+folder
        files = [entry for entry in os.listdir(path) if entry[-3:] == 'jpg']
        print("processing " + path)

        for file in files[:(200 if len(files) > 200 else len(files))]:
            originfile = path + "/" + file
            modfile = target + "/" + folder + "_" + file

            originIm = Image.open(originfile)
            cropped = ImageOps.fit(originIm, (96, 96), Image.ANTIALIAS)
            img_data = np.array(cropped.getdata(), np.uint8)
            if img_data.size == 27648:
                if counter % 7 == 0:
                    if len(dataTestX) < TEST_SIZE:
                        dataTestX.append(img_data.reshape(cropped.size[1], cropped.size[0], 3))
                else:
                    if len(dataTrainX) < TRAIN_SIZE:
                        dataTrainX.append(img_data.reshape(cropped.size[1], cropped.size[0], 3))
                counter += 1
            else:
                print 'Skipped file', file

dataTrainX = dataTrainX
dataTestX = dataTestX

save_zipped_pickle("MITTrainInstance.gz", dataTrainX)
save_zipped_pickle("MITTestInstance.gz", dataTestX)
