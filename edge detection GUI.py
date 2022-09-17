#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:07:13 2018

@author: dexter

"""
from tkinter import *
from PIL import Image, ImageOps
from PIL import ImageTk
from tkinter import filedialog
import cv2
import time
import random
import sys
import numpy as np


def feature_extractor(img):
    features_list = []
    r, c = img.shape
    for i in range(r-1):
        for j in range(c-1):
            d_list = []
            d_list.append(img[i, j+1]-img[i, j])
            d_list.append(img[i+1, j]-img[i, j])
            d_list.append(img[i+1, j+1]-img[i, j])
            features_list.append(d_list)
    return features_list

def predictor(img):
    import numpy as np
    from tensorflow.keras.models import load_model
    from PIL import ImageOps
    img = ImageOps.grayscale(img)
    img = np.array(img)
    r, c = img.shape
    savedModel = load_model("./edge detector.h5")
    predictions = savedModel.predict(np.array(feature_extractor(img)))
    predictions = predictions.reshape((r-1, c-1)) * 255
    predictions[predictions>72] = 255
    predictions[predictions<72] = 0
    return Image.fromarray(predictions)
    
    

x= random.randint(60,201)
def click_me1():
    time.sleep(3)
    action1.configure(text=" getting ")
    #a_label1.configure(foreground= "red")
    action1.destroy()
    root.geometry("300x300")
    w = Label(root, text="RESULT:", font=("Helvetica", 18),fg="black")
    w.place(x=50,y=50)
    if x>=60 and x<=110:
        w = Label(root, text="Image saved", font=("Helvetica", 16),fg="blue")
        w.place(x=100,y=150)
    elif x>110 and x<=126:
        w = Label(root, text="Image sent", font=("Helvetica", 16),fg="blue")
        w.place(x=100,y=150)
    else:
        w = Label(root, text="Image crashed ", font=("Helvetica", 16),fg="blue")
        w.place(x=100,y=150)
        
def select_image():
    global panelA, panelB,panelC
    path = filedialog.askopenfilename()  
    time.sleep(2)
    if len(path) > 0:
        image = Image.open(path)
        gray = ImageOps.grayscale(image)
        #for original image
        #image = Image.fromarray(image)
        img = image.resize((300,300))
        img = ImageTk.PhotoImage(img)
        
        edge_ann = predictor(gray)
        edge_ann = edge_ann.resize((300, 300))
        edge_ann = ImageTk.PhotoImage(edge_ann)
        



        
        if panelA is None or panelB is None or panelC is None:
            my_font=('times', 36, 'bold')
            panelA = Label(root,text="Original",width=300,font=my_font,image=img)
            panelA.image = img
            panelA.pack(side="left", padx=100, pady=100)
            #panelA.grid(row=4,column=1,columnspan=4,padx = 5, pady = 5)
           # panelB = Label(root,text="Sobel_Edge",image=sobel_edge)
            #panelB.image = sobel_edge
            #panelB.pack(side="right", padx=100, pady=100)
            #panelB.grid(row=4,column=8,columnspan=4,padx = 5, pady = 5)
            panelC = Label(root,text="Edge Detected ANN",image=edge_ann)#add custom edge detected function here
            panelC.image = edge_ann
            panelC.pack(side="left", padx=100, pady=100)

            #panelC.grid(row=1,column=12,columnspan=4,padx = 5, pady = 5)

            
            
            
        else:	
            panelA.configure(image=img)
            panelB.configure(image=edge_ann)
            panelA.image = img
            panelB.image = edge_ann
    btn.destroy()
         


#im = Image.open(r"C:\Users\manoj_9hatybv\OneDrive\Desktop\Academics\SEM-7\ML with Python\Mini Project\lena_full.jpg")
#roo= Tk()

#tkimage = ImageTk.PhotoImage(im)
#a=Label(roo, image=tkimage)
#a.pack()
#b= Label(roo, text="A TUTORIAL TO KNOW ABOUT TKINTER GUI AND EDGE DETECTION ",font=("Helvetica", 18),fg="red")
#b.place(x=10,y=300)
#roo.geometry("820x420")

root = Tk()
root.title("Edge Detector")
panelA = None
panelB = None
panelC = None
#root.geometry("1366x728")
root.geometry("1080x720")

action = Button(root,text= "Quit ",font=("Helvetica", 10),command=root.destroy)
action.pack(side="bottom")
my_font1=('times', 18, 'bold')
head = Label(root,text='Edge Detector by Revanth(BT19ECE015),Manikanta(BT19ECE068)',width=300,font=my_font1)
head.pack(side="top",padx=100, pady=10)
head2 = Label(root,text='& Manoj(BT19ECE064)',width=30,font=my_font1)
head2.pack(side="top",padx=100, pady=0)


#action.grid(row=9000,column=8000,columnspan=4,padx = 5, pady = 5)
#top names
#my_font1=('times', 18, 'bold')
#t = Label(root,text='Edge Detection by Manoj,Revanth,Manikanta',width=30,font=my_font1)  
#t.grid(row=1,column=1,columnspan=4)
#action1 = Button(root,text= "Get data",font=("Helvetica", 10),command=click_me1)
#action1.pack(side="bottom")
btn = Button(root, text="Select an image", command=select_image)
#btn.grid(row=2,column=1,columnspan=4)
btn.pack(side="top")
#btn.grid(row=1,column=8,columnspan=4)

# kick off the GUI
#roo.after(20,roo.destroy)
root.mainloop()
#roo.mainloop()


# In[ ]:





# In[ ]:




