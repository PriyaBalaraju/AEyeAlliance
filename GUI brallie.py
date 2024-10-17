from tkinter import *
import tkinter as tk
import cv2
import os
import math
from tkinter import filedialog
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile                            
import imutils
from PIL import Image
from PIL import ImageTk
from sklearn.model_selection import KFold
# global variables
import time
from PIL import ImageTk, Image
from skimage.filters import median
import pandas as pd
global rep
import csv
import copy
import random
from numpy import load
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle
from numpy import save
#from keras.utils import np_utils
import os
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
import pickle
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import keras
from sklearn.datasets import load_files
from tensorflow.python.keras.utils import np_utils
#from keras.utils import np_utils
import numpy as np
from glob import glob
import os
import cv2
import imutils
import os
import time
from glob import glob
from tensorflow.keras.preprocessing import image                  
from tqdm import tqdm
from gtts import gTTS
import os
from playsound import playsound



def classification(filterbank_features):
    clas1 = [item[8:-1] for item in sorted(glob("./dataa/*/"))]
    def path_to_tensor(img_path, width=224, height=224):
        print(img_path)
        img = image.load_img(img_path, target_size=(width, height))
        x = image.img_to_array(img)
        return np.expand_dims(x, axis=0)
    def paths_to_tensor(img_paths, width=224, height=224):
        list_of_tensors = [path_to_tensor(img_paths, width, height)]
        return np.vstack(list_of_tensors)
    from tensorflow.keras.models import load_model
    model = load_model('trained_model_CNN.h5')
    test_tensors = paths_to_tensor(filename)/255
    pred=model.predict(test_tensors)
    pred=np.argmax(pred);
    print('Given Braille Scirpt is  = '+clas1[int(pred)])
    res = clas1[int(pred)]
    messagebox.showinfo('Given Braille Scirpt is: ',res)
        

        
    
    



class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.config(bg="#00b3b3")
        
        # changing the title of our master widget
        
        self.master.title("BRAILLE SCRIPT RECOGNITION")
        
        self.pack(fill=BOTH, expand=1)
        
        w = tk.Label(root, 
		 text=" BRAILLE SCRIPT RECOGNITION",
		 fg = "#efffff",
		 bg = "#000404",
		 font = "verdana 20 bold",
                 width = 30)
        w.pack()
        w.place(x=400, y=5)

        # creating a button instance
        quitButton = Button(self,command=self.query, text="LOAD IMAGE",bg="#006666",fg="#efffff",activebackground="White",width=20)
        quitButton.place(x=550, y=160)
        quitButton = Button(self,command=self.preprocess,text="PREPROCESSING",bg="#006666",fg="#efffff",activebackground="White",width=20)
        quitButton.place(x=550, y=360)
        quitButton = Button(self,command=self.feature, text="FEATURE EXTRACTION",bg="#006666",fg="#efffff",activebackground="White",width=20)
        quitButton.place(x=550, y=560)
        quitButton = Button(self,command=self.classification,text="PREDICT",bg="#efffff",activebackground="White",fg="#2f3737",width=20)
        quitButton.place(x=800, y=260)

        load = Image.open("logo.jpeg")
        render = ImageTk.PhotoImage(load)

        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image1.image = render
        image1.place(x=350, y=80)

        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image2.image = render
        image2.place(x=350, y=280)

        image3=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=350, y=480)
        

#       Functions

    def query(self, event=None):
        global rep
        rep = filedialog.askopenfilenames()
        img = cv2.imread(rep[0])
        img = cv2.resize(img,(256,256))
        Input_img=img.copy()
        print(rep[0])
        self.from_array = Image.fromarray(cv2.resize(img,(200,200)))
        load = Image.open(rep[0])
        render = ImageTk.PhotoImage(load.resize((200,200)))
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image1.image = render
        image1.place(x=350, y=50)
        
    def close_window(): 
        Window.destroy()
        
    def preprocess(self, event=None):
        global rep
        img = cv2.imread(rep[0])
        img = cv2.resize(img,(256,256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.from_array = Image.fromarray(cv2.resize(hsv_img,(200,200)))
        render = ImageTk.PhotoImage(self.from_array)
        image3=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=350, y=250)
        render = ImageTk.PhotoImage(load.resize((200,200)))
        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image2.image = render
        image2.place(x=350, y=250)

        
    def feature(self, event=None):
        global rep
        img = cv2.imread(rep[0])
        img = cv2.resize(img,(256,256))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25,0,20])
        upper_green = np.array([100,255,255])
        mask = cv2.inRange(img, lower_green, upper_green)
        result = cv2.bitwise_and(img, img, mask=mask)
        img= cv2.resize(result,(256,256), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(cv2.resize(img,(200,200)))
        render = ImageTk.PhotoImage(self.from_array)
        image3=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=350, y=450)
    def classification(self, event=None):
        global T,rep
        clas1 = [item[8:-1] for item in sorted(glob("./dataa/*/"))]
        def path_to_tensor(img_path, width=224, height=224):
            print(img_path)
            img = image.load_img(img_path, target_size=(width, height))
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)
        def paths_to_tensor(img_paths, width=224, height=224):
            list_of_tensors = [path_to_tensor(img_paths, width, height)]
            return np.vstack(list_of_tensors)
        from tensorflow.keras.models import load_model
        model = load_model('trained_model_CNN.h5')
        main_img = cv2.imread(rep[0])
        hsv = cv2.cvtColor(main_img, cv2.COLOR_BGR2HSV)
        low_val = (0,0,0)
        high_val = (100,200,150)
        mask = cv2.inRange(hsv, low_val,high_val)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))
        result = cv2.bitwise_and(main_img, main_img,mask=mask)
        test_tensors = paths_to_tensor(rep[0])/255
        pred=model.predict(test_tensors)
        print('Given Braiile Script is : '+str(clas1[np.argmax(pred)]))
        res=clas1[np.argmax(pred)]
        global T
        tts = gTTS(res, lang='en')
        tts.save("message_audio.mp3")
         # Play the generated audio
        playsound("message_audio.mp3")

        
        if res == 'a':
            messagebox.showinfo('Braiile Recognition','Given Script is: a')
            print("a")
        elif res == 'b':
            messagebox.showinfo('Braiile Recognition','Given Script is: b')
            print("b")
        elif res == 'c':
            messagebox.showinfo('Braiile Recognition','Given Script is: c')
            print("c")
        elif res == 'd':
            messagebox.showinfo('Braiile Recognition','Given Script is: d')
            print("d")
        elif res == 'e':
            messagebox.showinfo('Braiile Recognition','Given Script is: e')
            print("e")
        elif res == 'f':
            messagebox.showinfo('Braiile Recognition','Given Script is: f')
            print("f")
        elif res == 'g':
            messagebox.showinfo('Braiile Recognition','Given Script is: g')
            print("g")
        elif res == 'h':
            messagebox.showinfo('Braiile Recognition','Given Script is: h')
            print("h")
        elif res == 'i':
            messagebox.showinfo('Braiile Recognition','Given Script is: i')
            print("i")
        elif res == 'j':
            messagebox.showinfo('Braiile Recognition','Given Script is: j')
            print("j")
        elif res == 'k':
            messagebox.showinfo('Braiile Recognition','Given Script is: k')
            print("k")
        elif res == 'l':
            messagebox.showinfo('Braiile Recognition','Given Script is: l')
            print("l")
        elif res == 'm':
            messagebox.showinfo('Braiile Recognition','Given Script is: m')
            print("m")
        elif res == 'n':
            messagebox.showinfo('Braiile Recognition','Given Script is: n')
            print("n")
        elif res == 'o':
            messagebox.showinfo('Braiile Recognition','Given Script is: o')
            print("o")
        elif res == 'p':
            messagebox.showinfo('alphabet','p')
            print("p")
        elif res == 'q':
            messagebox.showinfo('Braiile Recognition','Given Script is: q')
            print("q")
        elif res == 'r':
            messagebox.showinfo('Braiile Recognition','Given Script is: r')
            print("r")
        elif res == 's':
            messagebox.showinfo('Braiile Recognition','Given Script is: s')
            print("s")
        elif res == 't':
            messagebox.showinfo('Braiile Recognition','Given Script is: t')
            print("t")
        elif res == 'u':
            messagebox.showinfo('Braiile Recognition','Given Script is: u')
            print("u")
        elif res == 'v':
            messagebox.showinfo('Braiile Recognition','Given Script is: v')
            print("v")
        elif res == 'w':
            messagebox.showinfo('Braiile Recognition','Given Script is: w')
            print("w")
        elif res == 'x':
            messagebox.showinfo('Braiile Recognition','Given Script is: x')
            print("x")
        elif res == 'y':
            messagebox.showinfo('Braiile Recognition','Given Script is: y')
            print("y")
        elif res == 'z':
            messagebox.showinfo('Braiile Recognition','Given Script is: z')
            print("z")
        elif res == '0':
            messagebox.showinfo('Braiile Recognition','Given Script is: 0')
            print("0")
        elif res == '1':
            messagebox.showinfo('Braiile Recognition','Given Script is: 1')
            print("1")
        elif res == '2':
            messagebox.showinfo('Braiile Recognition','Given Script is: 2')
            print("2")
        elif res == '3':
            messagebox.showinfo('Braiile Recognition','Given Script is: 3')
            print("3")
        elif res == '4':
            messagebox.showinfo('Braiile Recognition','Given Script is: 4')
            print("4")
        elif res == '5':
            messagebox.showinfo('Braiile Recognition','Given Script is: 5')
            print("5")
        elif res == '6':
            messagebox.showinfo('Braiile Recognition','Given Script is: 6')
            print("6")
        elif res == '7':
            messagebox.showinfo('Braiile Recognition','Given Script is: 7')
            print("7")
        elif res == '8':
            messagebox.showinfo('Braiile Recognition','Given Script is: 8')
            print("8")
        elif res == '9':
            messagebox.showinfo('Braiile Recognition','Given Script is: 9')
            print("9")
        
        
            T = Text(self, height=10, width=20)
            T.place(x=800, y=300)
            T.insert(END,res)
                
root = Tk()
root.geometry("1400x720")
app = Window(root)
root.mainloop()

        
