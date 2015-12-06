import gzip
import os
from PIL import Image, ImageOps
import pdb
import numpy as np
import cPickle
#matplotlib inline
base = './UECFOOD100'
folders = [entry for entry in os.listdir(base) if (ord(entry[0])>47 and ord(entry[0])<58)]
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
    path = base+"/"+folder
    files = [entry for entry in os.listdir(path) if entry[-3:] == 'jpg']
    print("processing " + path)
    for file in files:
        originfile = path + "/" + file
        modfile = target + "/" + folder + "_" + file

        originIm = Image.open(originfile)
        cropped = ImageOps.fit(originIm, (96, 96), Image.ANTIALIAS)
        if counter % 7 == 0:
            dataTestX.append(np.array(cropped.getdata(), np.uint8).reshape(cropped.size[1], cropped.size[0], 3))
            dataTestY.append(np.array([1]))
        else:
            dataTrainX.append(np.array(cropped.getdata(), np.uint8).reshape(cropped.size[1], cropped.size[0], 3))
            dataTrainY.append(np.array([1]))

        counter += 1

save_zipped_pickle("UECTrainInstance.gz", [dataTrainX, dataTrainY])
save_zipped_pickle("UECTestInstance.gz", [dataTestX, dataTestY])
