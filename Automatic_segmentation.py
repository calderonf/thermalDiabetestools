import numpy as np
import torch
import cv2
import os
import sys
from datetime import datetime
import flir_image_extractor as flirimageextractor
import Utils as u

import matplotlib
matplotlib.use('TKAgg') # Needed to have figures display properly. 
import matplotlib.pyplot as plt

 
def obtener_nombres_hdf5(nombre_archivo):
    """obtener_nombres_hdf5

    Args:
        nombre_archivo (_type_): nombre de archivo de entrada, 

    Returns:
        t_right_,       nombre de archivo de transfomracion del pie derecho
        t_left_,        nombre de archivo de transfomracion del pie izquierdo
        temp_right_,    nombre de archivo de temeratura del pie derecho
        temp_left_,     nombre de archivo de temeratura del pie izquierdo
        mask_right_,    nombre de archivo de mascara del pie derecho
        mask_left_,     nombre de archivo de mascara del pie izquierdo
        color_right_,   nombre de archivo de mascara del pie derecho
        color_left_,    nombre de archivo de mascara del pie izquierdo
        Raw_mask,       nombre de archivo de mascara de la imagen
        Raw_RGB,        nombre de archivo de rgb de la imagen
        Raw_temp,       nombre de archivo de temperaturas de la imagen
        
    """
    partes = nombre_archivo.split("_")  # Dividir el nombre de archivo en partes utilizando el guión bajo como separador
    fecha_hora = partes[1]  # Obtener la parte de fecha y hora
    
    t_right_ = "tr_right_" + fecha_hora.replace(".jpg", ".hdf5")  # Reemplazar la extensión .jpg por .hdf5
    t_left_ = "tr_left_" + fecha_hora.replace(".jpg", ".hdf5")  # Reemplazar la extensión .jpg por .hdf5
    
    temp_right_ = "temp_right_" + fecha_hora.replace(".jpg", ".csv")  # Reemplazar la extensión .jpg por .csv
    temp_left_ = "temp_left_" + fecha_hora.replace(".jpg", ".csv")  # Reemplazar la extensión .jpg por .csv
    
    mask_right_ ="mask_right_" + fecha_hora.replace(".jpg", ".png")  # Reemplazar la extensión .jpg por .csv
    mask_left_ ="mask_left_" + fecha_hora.replace(".jpg", ".png")  # Reemplazar la extensión .jpg por .csv
    
    color_right_ ="color_right_" + fecha_hora.replace(".jpg", ".png")  # Reemplazar la extensión .jpg por .csv
    color_left_ ="color_left_" + fecha_hora.replace(".jpg", ".png")  # Reemplazar la extensión .jpg por .csv
    
    Raw_mask ="Raw_mask" + fecha_hora.replace(".jpg", ".png")  # Reemplazar la extensión .jpg por .csv
    Raw_RGB ="Raw_RGB" + fecha_hora.replace(".jpg", ".png")  # Reemplazar la extensión .jpg por .csv
    Raw_temp ="Raw_temp" + fecha_hora.replace(".jpg", ".csv")  # Reemplazar la extensión .jpg por .csv
    
    return t_right_,t_left_,temp_right_,temp_left_, mask_right_, mask_left_, color_right_, color_left_,Raw_mask,Raw_RGB,Raw_temp


sys.path.append("C:/Users/Francisco/Dropbox/PC/Documents/GitHub/segment-anything")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "C:/Users/Francisco/Documents/GitHub/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)



class MaskSelector:
    def __init__(self, anns):
        self.anns = sorted(anns, key=lambda x: x['area'], reverse=True)
        self.final_mask = np.zeros_like(self.anns[0]['segmentation'], dtype=bool)
        print("otro punto en mascara final: ",self.final_mask[0, 0])
        
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.show_masks()

    def show_masks(self):
        img = np.ones((self.anns[0]['segmentation'].shape[0], self.anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in self.anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        self.ax.imshow(img)
        plt.show()

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            for ann in self.anns:
                if ann['segmentation'][y, x] and not (ann['segmentation'][0,0]):
                    print("Entro")
                    self.final_mask[ann['segmentation']]=True
                    break
                    
            #self.ax.clear()
            #self.ax.imshow(self.final_mask, cmap='gray')

    def get_final_mask(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        plt.close(self.fig)
        return self.final_mask

def select_and_merge_masks(anns):
    selector = MaskSelector(anns)
    plt.waitforbuttonpress()
    return selector.get_final_mask()

def visualize_and_save_final_mask(mask, save_path=None):
    # Visualizar la máscara final
    plt.imshow(mask, cmap='gray')
    plt.title("Final Combined Mask")
    plt.show()
    
    # Guardar la máscara final si se proporciona una ruta
    if save_path:
        plt.imsave(save_path, mask.astype(np.uint8) * 255, cmap='gray')


# Uso:
# anns = SamAutomaticMaskGenerator.generate(IMAGEN)
# final_mask = select_and_merge_masks(anns_sample)
# visualize_and_save_final_mask(final_mask)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=1.2*2*2*2*2*2*2*70*100,  # Requires open-cv to run post-processing
)

DEBUG=True
DEBUG_VISUAL=False
DEBUG_VISUAL_L2=False
PREVSIZE=(205, 443)
base_folder = r"C:\Users\Francisco\Documents\ThermalDataset"
thermal_images_folder="Thermal-Images"
flirfolder="flir"
sub_folders = ["Control", "Diabetic"]
patient_numbers = {
    "Control": [3, 4, 5, 6, 9, 11, 12, 13, 14, 16, 17, 18, 19],
    "Diabetic": [1, 2, 7, 8, 10, 20, 21, 22]
}
if DEBUG_VISUAL_L2:
    # Crear la figura y los subplots una vez antes del bucle
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    # Configurar los títulos de los subplots
    titles = ['fixed_imagenr', 'moving_imagenr', 'composition_imager', 'thermalright',
            'fixed_imagenl', 'moving_imagenl', 'composition_imagel', 'thermalleft']

    # Inicializar las imágenes con datos vacíos
    images = [None] * 8

flir_fixed = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Windows\\exiftool.exe")
flir_moving = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Windows\\exiftool.exe")

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
                if DEBUG:
                    # Imprimir la lista de imágenes
                    for image_path,file_name, image_datetime in image_list:
                        print(image_path, file_name, image_datetime)
                #Acá inicia el procesamiento de las imagenes de la lista 
                    for index, item in enumerate(image_list[0:]):
                        if DEBUG:
                            print("processing image: ", item[1] )
                        flir_moving.process_image(item[0])
                        _, _,temp_moving,image_moving = u.extract_images(flir_moving,plot=0)
                        
                            
                        filer,filel,filethermalr,filethermall, mask_right_, mask_left_, color_right_, color_left_,Raw_mask,Raw_RGB,Raw_temp=obtener_nombres_hdf5(item[1])
                        
                        #cv2.imwrite(str(os.path.join(folder_path, Raw_mask)),binary_mask)
                        
                        #cv2.imwrite(str(os.path.join(folder_path, Raw_RGB)),cv2.cvtColor(image_moving, cv2.COLOR_RGB2BGR))
                        #u.save_thermal_csv(temp_moving,str(os.path.join(folder_path, Raw_temp)))
                        masks2 = mask_generator_2.generate(image_moving)
                        final_mask = select_and_merge_masks(masks2)
                        visualize_and_save_final_mask(final_mask,save_path=str(os.path.join(folder_path, Raw_mask)))
                        """
                        print(type(masks2))
                        print(dir(masks2))
                        print(type(masks2[0]))
                        print(dir(masks2[0]))
                        print(masks2[0].keys())
                        print(masks2[0].keys())
                        print('segmentation ',type(masks2[0]['segmentation']))
                        print('segmentation ',masks2[0]['segmentation'].shape)
                        print('area ',type(masks2[0]['area']))
                        print('bbox ',type(masks2[0]['bbox']))
                        print('predicted_iou ',type(masks2[0]['predicted_iou']))
                        print('point_coords ',type(masks2[0]['point_coords']))
                        print('stability_score ',type(masks2[0]['stability_score']))
                        print('crop_box ',type(masks2[0]['crop_box']))
                        import pickle
                        with open("data.pkl", "wb") as file:
                            pickle.dump(masks2, file)
                        """
            else:
                print("Error, no encuentro el directorio: ",folder_path)
