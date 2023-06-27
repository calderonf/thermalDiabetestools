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
DEBUG=False
DEBUG_VISUAL=False
DEBUG_VISUAL_L2=True
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
                #Acá inicia el procesamiento d ela primera imagen de la lista con respecto a las demás. 
                flir_fixed.process_image(image_list[0][0], RGB=True)
                _, _,temp_fixed,image_fixed = u.extract_images(flir_fixed,plot=0)
                segmented_Feet_fixed,segmented_temps_fixed,segmented_masks_fixed,binary_mask=u.Find_feets(image_fixed,temp_fixed,percentage=20)
                
                filer,filel,filethermalr,filethermall,mask_right_, mask_left_, color_right_, color_left_, Raw_mask, Raw_RGB, Raw_temp=obtener_nombres_hdf5(image_list[0][1])
                
                u.save_thermal_csv(segmented_temps_fixed[0],str(os.path.join(folder_path, filethermalr)))
                u.save_thermal_csv(segmented_temps_fixed[1],str(os.path.join(folder_path, filethermall)))
                
                cv2.imwrite(str(os.path.join(folder_path, mask_right_)),segmented_masks_fixed[0])
                cv2.imwrite(str(os.path.join(folder_path, mask_left_)),segmented_masks_fixed[1])
                
                cv2.imwrite(str(os.path.join(folder_path, color_right_)),cv2.cvtColor(segmented_Feet_fixed[0], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(os.path.join(folder_path, color_left_)),cv2.cvtColor(segmented_Feet_fixed[1], cv2.COLOR_RGB2BGR))
                
                cv2.imwrite(str(os.path.join(folder_path, Raw_mask)),binary_mask)
                cv2.imwrite(str(os.path.join(folder_path, Raw_RGB)),cv2.cvtColor(image_fixed, cv2.COLOR_RGB2BGR))
                u.save_thermal_csv(temp_fixed,str(os.path.join(folder_path, Raw_temp)))
                
                
                if DEBUG_VISUAL:
                    print("showing images")
                    u.plot_images_and_thermal(segmented_Feet_fixed,segmented_temps_fixed)
                    
                for index, item in enumerate(image_list[1:]):
                    if DEBUG:
                        print("we are going to make the registration between: \n",image_list[0][1], "and \n", item[1] )
                    flir_moving.process_image(item[0], RGB=True)
                    _, _,temp_moving,image_moving = u.extract_images(flir_moving,plot=0)
                    segmented_Feet_moving,segmented_temps_moving,segmented_masks_moving,binary_mask=u.Find_feets(image_moving,temp_moving,percentage=20)
                    if DEBUG_VISUAL:
                        print("showing images")
                        u.plot_images_and_thermal(segmented_Feet_moving,segmented_temps_moving)
                    filer,filel,filethermalr,filethermall, mask_right_, mask_left_, color_right_, color_left_,Raw_mask,Raw_RGB,Raw_temp=obtener_nombres_hdf5(item[1])
                    derecho =sr.register_images_Similarity2DTransform(segmented_Feet_fixed[0],segmented_Feet_moving[0],os.path.join(folder_path, filer))
                    thermalright= sr.cv2_grid_sampling(segmented_temps_fixed[0], segmented_temps_moving[0], derecho["Transform"], is_binary=False)
                    izquierdo =sr.register_images_Similarity2DTransform(segmented_Feet_fixed[1],segmented_Feet_moving[1],os.path.join(folder_path, filel))
                    thermalleft= sr.cv2_grid_sampling(segmented_temps_fixed[1], segmented_temps_moving[1], izquierdo["Transform"], is_binary=False)
                    
                    binaryright= sr.cv2_grid_sampling(segmented_masks_fixed[0], segmented_masks_moving[0], derecho["Transform"], is_binary=True)
                    
                    binaryleft= sr.cv2_grid_sampling(segmented_masks_fixed[1], segmented_masks_moving[1], izquierdo["Transform"], is_binary=True)
                    
                    u.save_thermal_csv(thermalright,os.path.join(folder_path, filethermalr))
                    u.save_thermal_csv(thermalleft,os.path.join(folder_path, filethermall))
                    
                    cv2.imwrite(str(os.path.join(folder_path, mask_right_)),binaryright)
                    cv2.imwrite(str(os.path.join(folder_path, mask_left_)),binaryleft)
                    
                    cv2.imwrite(str(os.path.join(folder_path, color_right_)),cv2.cvtColor(segmented_Feet_moving[0], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(os.path.join(folder_path, color_left_)),cv2.cvtColor(segmented_Feet_moving[1], cv2.COLOR_RGB2BGR))
                    
                    cv2.imwrite(str(os.path.join(folder_path, Raw_mask)),binary_mask)
                    cv2.imwrite(str(os.path.join(folder_path, Raw_RGB)),cv2.cvtColor(image_moving, cv2.COLOR_RGB2BGR))
                    u.save_thermal_csv(temp_moving,str(os.path.join(folder_path, Raw_temp)))
                    
                    if DEBUG_VISUAL_L2:
                        fixed_imagenr = cv2.resize(sr.sitk_image_to_opencv(derecho["fixed"]), PREVSIZE)
                        moving_imagenr = cv2.resize(sr.sitk_image_to_opencv(derecho["moving"]), PREVSIZE)
                        composition_imager = cv2.resize(sr.sitk_image_to_opencv(derecho["composition"]), PREVSIZE)
                        fixed_imagenl = cv2.resize(sr.sitk_image_to_opencv(izquierdo["fixed"]), PREVSIZE)
                        moving_imagenl = cv2.resize(sr.sitk_image_to_opencv(izquierdo["moving"]), PREVSIZE)
                        composition_imagel = cv2.resize(sr.sitk_image_to_opencv(izquierdo["composition"]), PREVSIZE)
                        thermalrightcv  = cv2.resize(thermalright, PREVSIZE)
                        thermalleftcv    = cv2.resize(thermalleft, PREVSIZE)              
                        data = [fixed_imagenr, moving_imagenr, composition_imager, thermalrightcv,
                                fixed_imagenl, moving_imagenl, composition_imagel, thermalleftcv]
                        if images[0] is None:
                            # Crear las imágenes en la primera iteración
                            for j, ax in enumerate(axes.flat):
                                images[j] = ax.imshow(data[j])
                                ax.set_title(titles[j])
                        else:
                            # Actualizar los datos de las imágenes existentes
                            for j in range(8):
                                images[j].set_data(data[j])
                            # Ajustar manualmente los límites de color para cada imagen
                            if j in [0, 1, 2, 4, 5, 6]:
                                images[j].set_clim(np.min(data[j][:, :, 0]), np.max(data[j][:, :, 0]))
                            else:
                                images[j].set_clim(np.min(data[j]), np.max(data[j]))
                        # Redibujar la figura
                        fig.canvas.draw()
                        # Pausa brevemente para permitir la actualización de la figura
                        plt.pause(0.1)
            else:
                print("Error, no encuentro el directorio: ",folder_path)

