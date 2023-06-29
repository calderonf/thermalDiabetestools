
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
from scipy.interpolate import interp1d

DEBUG=True
DOC=True


def calcular_derivada_temporal(imagenes, tiempos, coeficientes):
    # Convertir la lista de imágenes en un arreglo tridimensional (altura x ancho x número de imágenes)
    imagenes_array = np.array(imagenes)
    
    # Verificar que las listas de imágenes y tiempos tengan la misma longitud
    if len(imagenes) != len(tiempos):
        raise ValueError("Las listas de imágenes y tiempos deben tener la misma longitud.")
    
    # Obtener las dimensiones de las imágenes
    altura, ancho, num_imagenes = imagenes_array.shape
    
    # Verificar que los coeficientes sean válidos
    if len(coeficientes) < 2:
        raise ValueError("Los coeficientes deben tener una longitud de al menos 2.")
    
    # Asegurarse de que los coeficientes sumen 0 para mantener el nivel de gris promedio
    coeficientes = np.array(coeficientes)
    coeficientes = coeficientes - np.mean(coeficientes)
    
    # Crear un arreglo para almacenar las imágenes de salida
    imagenes_salida = []
    
    # Calcular la derivada temporal teniendo en cuenta los tiempos
    for i in range(1, num_imagenes):
        # Calcular el intervalo de tiempo entre las imágenes consecutivas
        dt = tiempos[i] - tiempos[i-1]
        
        # Interpolar los valores de las imágenes en los puntos intermedios
        interpolador = interp1d([tiempos[i-1], tiempos[i]], [imagenes_array[:, :, i-1], imagenes_array[:, :, i]], axis=0)
        imagenes_interpoladas = interpolador(tiempos[i-1] + dt/2)
        
        # Calcular la derivada temporal mediante la convolución
        derivada = np.sum(coeficientes[:, None, None] * (imagenes_interpoladas - imagenes_array[:, :, i-1]), axis=0) / dt
        imagenes_salida.append(derivada)
    
    return imagenes_salida

def calcular_diferencia_segundos(datetime1, datetime2):
    diferencia = datetime2 - datetime1
    segundos = diferencia.total_seconds()
    return int(segundos)

def main():
    if DOC:
        print(__doc__)
    base_folder = r"C:\Users\Francisco\Documents\ThermalDataset"
    thermal_images_folder="Thermal-Images"
    flirfolder="flir"
    sub_folders = ["Control", "Diabetic"]
    patient_numbers = {
        "Control": [3, 5, 9, 6, 4, 11, 12, 13, 14, 16, 17, 18, 19],
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
                    for i in range(len(image_list) - 1):
                        
                        image_path1,file_name1, image_datetime1 = image_list[i]
                        image_path2,file_name2, image_datetime2 = image_list[i + 1]
                        print(image_path1, file_name1, image_datetime1)
                        print(image_path2, file_name2, image_datetime2)
                        
                        flir_fixed.process_image(image_path1, RGB=True)
                        flir_fixed.process_image(image_path2, RGB=True)
                        _, _,temp_fixed,image_fixed = u.extract_images(flir_fixed,plot=0)
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        

if __name__ == '__main__':
    main()