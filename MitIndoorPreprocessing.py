import os
from PIL import Image, ImageOps
import cv2
import numpy as np
from pylearn2.utils import serial

base = './MIT'
excetionList = ["bakery","bar","buffet","casino","deli","dining_room","fastfood_restaurant",
                "grocerystore","pantry","restaurant","restaurant_kitchen","winecellar"] 
folders = [entry for entry in os.scandir(base)]

target = './Food'
dataTrainX=[]
dataTestX=[]
counter = 0
for folder in folders:
    if folder.name not in excetionList:
        
        path = base+"/"+folder.name
        files = [entry for entry in os.scandir(path) if entry.name[-3:]=='jpg']
        print("processing "+ path)
        for file in files:
            originfile = path+"/"+file.name
            modfile = target+"/"+folder.name+"_"+file.name

            originIm = Image.open(originfile)
            cropped = ImageOps.fit(originIm,(96, 96),Image.ANTIALIAS)

            if counter%7==0:
                dataTestX.append(np.array(cropped.getdata(),np.uint8).reshape(cropped.size[1], cropped.size[0], 3))
                dataTestY.append(0)
            else:
                dataTrainX.append(np.array(cropped.getdata(),np.uint8).reshape(cropped.size[1], cropped.size[0], 3))
                dataTrainY.append(0)

            counter +=1
        
serial.save("MITTrainInstance.pkl",dataTrainX)
serial.save("MITTestInstance.pkl",dataTestX)
serial.save("MITTrainInstanceY.pkl",dataTrainY)
serial.save("MITTestInstanceY.pkl",dataTestY)