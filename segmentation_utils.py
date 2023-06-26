
# Python 2/3 compatibility
from __future__ import print_function
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg') # Needed to have figures display properly. 
import flirimageextractor
import Utils as u
import cv2
import registration_simpleitk as sr
import numpy as np
import cv2
import sys


#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using the
right mouse button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' to update the output.

Key '1' - To select areas of sure background
Key '2' - To select areas of sure foreground
Key '3' - To select areas of probable background
Key '4' - To select areas of probable foreground

Key 'e' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

class GrabCutApp():
    BLUE = [255,0,0]        # rectangle color
    RED = [0,0,255]         # PR BG
    GREEN = [0,255,0]       # PR FG
    BLACK = [0,0,0]         # sure BG
    WHITE = [255,255,255]   # sure FG

    DRAW_BG = {'color' : BLACK, 'val' : 0}
    DRAW_FG = {'color' : WHITE, 'val' : 1}
    DRAW_PR_BG = {'color' : RED, 'val' : 2}
    DRAW_PR_FG = {'color' : GREEN, 'val' : 3}

    # setting up flags
    rect = (0,0,1,1)
    drawing = False         # flag for drawing curves
    rectangle = False       # flag for drawing rect
    rect_over = False       # flag to check if rect drawn
    rect_or_mask = 100      # flag for selecting rect or mask mode
    value = DRAW_FG         # drawing initialized to FG
    thickness = 10           # brush thickness
        
    def __init__(self, imagen):
        self.img = imagen
        

    def onmouse(self, event, x, y, flags, param):
        # Draw Rectangle
        if event == cv2.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.rect_or_mask = 0

        elif event == cv2.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # draw touchup curves

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv2.circle(self.img, (x,y), self.thickness, self.value['color'], -1)
                cv2.circle(self.mask, (x,y), self.thickness, self.value['val'], -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv2.circle(self.img, (x, y), self.thickness, self.value['color'], -1)
                cv2.circle(self.mask, (x, y), self.thickness, self.value['val'], -1)

    def run(self):
        self.img2 = self.img.copy()                               # a copy of original image
        self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
        self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown

        # input and output windows
        cv2.namedWindow('output')
        cv2.namedWindow('input')
        cv2.setMouseCallback('input', self.onmouse)
        cv2.moveWindow('input', self.img.shape[1]+10,90)

        print(" Instructions: \n")
        print(" Draw a rectangle around the object using right mouse button \n")

        while(1):

            cv2.imshow('output', self.output)
            cv2.imshow('input', self.img)
            k = cv2.waitKey(1)

            # key bindings
            if k == 27:         # esc to exit
                break
            elif k == ord('1'): # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = self.DRAW_BG
            elif k == ord('2'): # FG drawing
                print(" mark foreground regions with left mouse button \n")
                self.value = self.DRAW_FG
            elif k == ord('3'): # PR_BG drawing
                self.value = self.DRAW_PR_BG
            elif k == ord('4'): # PR_FG drawing
                self.value = self.DRAW_PR_FG
            elif k == ord('s'): # save image
                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                res = np.hstack((self.img2, bar, self.img, bar, self.output))
                cv2.imwrite('grabcut_output.png', res)
                print(" Result saved as image \n")
            elif k == ord('r'): # reset everything
                print("resetting \n")
                self.rect = (0,0,1,1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                self.img = self.img2.copy()
                self.mask = np.zeros(self.img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
                self.output = np.zeros(self.img.shape, np.uint8)           # output image to be shown
            elif k == ord('e'): # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                try:
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    if (self.rect_or_mask == 0):         # grabcut with rect
                        cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                        self.rect_or_mask = 1
                    elif (self.rect_or_mask == 1):       # grabcut with mask
                        cv2.grabCut(self.img2, self.mask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
                except:
                    import traceback
                    traceback.print_exc()

            mask2 = np.where((self.mask==1) + (self.mask==3), 255, 0).astype('uint8')
            self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask2)
        print('Done')
        return mask2

"""
if __name__ == '__main__':
    print(__doc__)
    App().run()
    cv2.destroyAllWindows()
"""
DEBUG=True
DOC=True

def main():
    if DOC:
        print(__doc__)
    base_folder = r"C:\Users\Francisco\Documents\ThermalDataset"
    thermal_images_folder="Thermal-Images"
    flirfolder="flir"
    sub_folders = ["Control", "Diabetic"]
    patient_numbers = {
        "Control": [9, 3, 5, 6, 4, 11, 12, 13, 14, 16, 17, 18, 19],
        "Diabetic": [1, 2, 7, 8, 10, 20, 21, 22]
    }
    flir_fixed = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Windows\\exiftool.exe")

    
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(base_folder, sub_folder,thermal_images_folder )
        for patient_number in patient_numbers[sub_folder]:
            patient_folder = os.path.join(sub_folder_path, f"paciente_{patient_number}")
            flir_folder = os.path.join(patient_folder, flirfolder)
            for folder_name in ["control", "dimple", "stand_up"]:#, "reposo"
                folder_path = os.path.join(flir_folder, folder_name)
                if os.path.exists(folder_path):
                    image_list = []
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith(".jpg"):
                            file_path = os.path.join(folder_path, file_name)
                            file_datetime = datetime.strptime(file_name[5:20], "%Y%m%dT%H%M%S")
                            image_list.append((file_path, file_name, file_datetime))
                    # Ordenar la lista de imágenes por fecha
                    image_list = sorted(image_list, key=lambda x: x[2])# x[2] is file_datetime
                        
                    #Acá inicia el procesamiento de la primera imagen de la lista con respecto a las demás. 
                    for image_path,file_name, image_datetime in image_list:
                        
                        print(image_path, file_name, image_datetime)
                        flir_fixed.process_image(image_path, RGB=True)
                        _, _,temp_fixed,image_fixed = u.extract_images(flir_fixed,plot=0)
                        image_fixed_brg=cv2.cvtColor(image_fixed, cv2.COLOR_RGB2BGR)
                        gcapp = GrabCutApp(image_fixed_brg)
                        imagen_binaria=gcapp.run()
                        bin_out=u.Refine_Detection_of_feet(imagen_binaria,image_fixed_brg)
                        
                        cv2.imwrite(image_path.replace("flir_","mask_").replace(".jpg",".png"), bin_out)

if __name__ == '__main__':
    main()