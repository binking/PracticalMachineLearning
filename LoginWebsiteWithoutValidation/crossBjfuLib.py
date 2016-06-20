#!/usr/bin/env python
# coding: utf-8
# Excute under Jupyter Notebook

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mg
import numpy as np
import requests
from IPython.display import Image

rs = requests.session() # initialize a session, including validation pictures cathching and post basci info to server
valCode = rs.get('http://202.204.121.41:8080/reader/captcha.php', stream=True, verify=False)
fr = open('valCode.png', 'wb')  # Taking this png as an example
fr.write(valCode.raw.read())
fr.close()

Image('valCode.png') # In IPython, it will show the png

# All the info of sheet can be found in reader/redr_verify.php
# --- using Chrome --> Developers' tools --> network --> doc --> scroll down
sheet_info = {"number":"131114102", # My number
             "passwd":"********",  # Passwd can't be shared
             "select":"cert_no",
             "captcha":"****"} # captcha is what you see, but I plan to use training valition images set using knn algo or logistic regression
res = rs.post('http://202.204.121.41:8080/reader/redr_verify.php', data=sheet_info, verify=False) # Our breakthrough

res.encoding = 'utf-8'
print res.text  # Check the html

rate = [1, 14, 27, 40, 53] # Split one image(60*36*4) into 4 part(14*13*4)

img = mg.imread('./Validation_Examples/3.gif')
plt.subplot(2,2,1)
plt.imshow(img[:, rate[0]:rate[1], :])
plt.subplot(2,2,2)
plt.imshow(img[:, rate[1]:rate[2], :])
plt.subplot(2,2,3)
plt.imshow(img[:, rate[2]:rate[3], :])
plt.subplot(2,2,4)
plt.imshow(img[:, rate[3]:rate[4], :])

def gif2gray(gif): # Convert gif/rgb to greyscale
    return np.dot(gif[...,:3], [0.299, 0.587, 0.114])
subimg = img[14:28, rate[0]:rate[1], :] # testing
plt.imshow(gif2gray(subimg), cmap="Greys_r") # Remember set cmap, otherwise it will show awful figure
# Save image : plt.imsave()