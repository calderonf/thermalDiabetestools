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


class GrabCutApp:
    def __init__(self):
        self.rect = None
        self.drawing = False
        self.image = None
        self.mask = None
        self.output = None
        self.init_rect = False

    def set_image_and_win_name(self, image, win_name):
        self.image = image
        self.win_name = win_name

    def show_image(self):
        cv2.imshow(self.win_name, self.image)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect = (x, y, 1, 1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect = (self.rect[0], self.rect[1], x - self.rect[0], y - self.rect[1])
            self.init_rect = True
            self.process_image()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.rect = (self.rect[0], self.rect[1], x - self.rect[0], y - self.rect[1])
                temp_output = self.image.copy()
                cv2.rectangle(temp_output, (self.rect[0], self.rect[1]), (self.rect[0] + self.rect[2], self.rect[1] + self.rect[3]), (0, 255, 0), 2)
                cv2.imshow(self.win_name, temp_output)

    def reset(self):
        self.rect = None
        self.drawing = False
        self.init_rect = False
        self.mask = None
        self.output = None

    def next_iter(self):
        if self.init_rect:
            self.mask = self.process_grab_cut()
            self.output = self.apply_mask()
            return 1
        else:
            return 0
        
    def process_image(self):
        if self.init_rect:
            self.mask = self.process_grab_cut()
            self.output = self.apply_mask()
            
    def process_grab_cut(self):
        mask = cv2.GC_BGD * np.ones(self.image.shape[:2], dtype=np.uint8)
        mask[self.rect[1]:self.rect[1] + self.rect[3], self.rect[0]:self.rect[0] + self.rect[2]] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        cv2.grabCut(self.image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        return mask

    def apply_mask(self):
        output = np.where((self.mask == cv2.GC_PR_FGD) | (self.mask == cv2.GC_FGD), 255, 0).astype(np.uint8)
        result = cv2.bitwise_and(self.image, self.image, mask=output)
        return result


def GrabcutToImage(image):

    win_name = "image"
    gc_app = GrabCutApp()
    gc_app.set_image_and_win_name(image, win_name)
    gc_app.show_image()
    
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)


    cv2.setMouseCallback(win_name, gc_app.on_mouse)
    while True:
        key = cv2.waitKey(0)
        if key == 27:
            print("Exiting ...")
            break
        elif key == ord('r'):
            print()
            gc_app.reset()
            gc_app.show_image()
        elif key == ord('n'):
            iter_count = gc_app.next_iter()
            print(f"<{iter_count}... ", end='')
            if iter_count > 0:
                gc_app.show_image()
                print(f"{iter_count}>")
            else:
                print("rect must be determined>")

    cv2.destroyWindow(win_name)
    return gc_app.apply_mask()
            
DEBUG=True


def main():

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
                        imagen_binaria = GrabcutToImage(cv2.cvtColor(image_fixed, cv2.COLOR_RGB2BGR))
                            # Mostrar la imagen binaria resultante
                        cv2.imshow('Imagen Binaria', imagen_binaria )
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()