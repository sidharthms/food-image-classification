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

dataTestX = []
dataTestY = []

dataTrainX = []
dataTrainY = []

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
            if cropped.size == (96, 96, 3):
                if counter % 7 == 0:
                    dataTestX.append(np.array(cropped.getdata(), np.uint8).reshape(cropped.size[1], cropped.size[0], 3))
                    dataTestY.append(np.array([0]))
                else:
                    dataTrainX.append(np.array(cropped.getdata(), np.uint8).reshape(cropped.size[1],
                                                                                    cropped.size[0], 3))
                    dataTrainY.append(np.array([0]))

                counter += 1

dataTrainX = dataTrainX[:8000]
dataTrainY = dataTrainY[:8000]
dataTestX = dataTestX[:2000]
dataTestY = dataTestY[:2000]

save_zipped_pickle("MITTrainInstance.gz", [dataTrainX, dataTrainY])
save_zipped_pickle("MITTestInstance.gz", [dataTestX, dataTestY])
