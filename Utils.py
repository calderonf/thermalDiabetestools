# Importing Functions
import flirimageextractor
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import subprocess
import cv2
import glob
import matplotlib
matplotlib.use('TKAgg') # Needed to have figures display properly. 

def reorder_contours(contornos):
    # Calcula los centroides de los contornos
    centroides = []
    for contorno in contornos:
        momento = cv2.moments(contorno)
        cx = int(momento["m10"] / momento["m00"])
        #cy = int(momento["m01"] / momento["m00"])
        centroides.append(cx)

    # Ordena los contornos según la posición de sus centroides en el eje x
    contornos_ordenados = [c for _, c in sorted(zip(centroides, contornos), key=lambda pair: pair[0])]
    return contornos_ordenados

def segment_skin(image):
    """
    Segment the skin color in an image to identify the foot region.

    Args:
        image (numpy.ndarray): Input image as a numpy array.

    Returns:
        numpy.ndarray: Binary mask with the identified foot region.

    """
    lower = np.array([0, 30, 40], dtype = "uint8")
    upper = np.array([20, 180, 255], dtype = "uint8")
    lower2 = np.array([172, 30, 40], dtype = "uint8")
    upper2 = np.array([180, 180, 210], dtype = "uint8")
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    skinMask2 = cv2.inRange(converted, lower2, upper2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skinMask2 = cv2.GaussianBlur(skinMask2, (3, 3), 0)
    skin1 = cv2.bitwise_and(image, image, mask = skinMask)
    skin2 = cv2.bitwise_and(image, image, mask = skinMask2)
    skin = cv2.bitwise_or(skin1,skin2)
    return skin

def fethearing_bin_to_color(binary_image, color_image):
    # Operación de cierre morfológico con un kernel elíptico de tamaño 5x5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_image1 = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    kerne2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_image = cv2.morphologyEx(closed_image1, cv2.MORPH_CLOSE, kerne2)
    
    # Fethearing usando filtro guiado
    filtered_image = cv2.ximgproc.guidedFilter(color_image, closed_image, 50, eps=1e-4)#5, eps=1e-4
    
    # Umbralización de la imagen filtrada
    _, thresholded_image = cv2.threshold(filtered_image,20, 255, cv2.THRESH_BINARY)
    
    kerne3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    output_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kerne3)
    
    return filtered_image,output_image

def draw_largest_contours(image):
    # Hacer una AND entre los 3 canales de la imagen
    and_image = np.bitwise_and(image[:,:,0], np.bitwise_and(image[:,:,1], image[:,:,2]))

    # Encontrar los contornos más externos
    contours, _ = cv2.findContours(and_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos por su área de mayor a menor
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Encontrar los dos contornos más grandes
    largest_contours = contours[:2]
    
    # Crear una nueva imagen binaria con solo los contornos más grandes dibujados
    new_image = np.zeros_like(image)
    cv2.drawContours(new_image, largest_contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # Convertir la imagen a binaria
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    _, new_image = cv2.threshold(new_image, 10, 255, cv2.THRESH_BINARY)

    return new_image

def resize_contour_boxes_rotated(binary_image, color_image, thermal_image, percentage):
    contours_no_order, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours=reorder_contours(contours_no_order)
    
    #print("tamaño contornos:",len (contours))
    rects = [cv2.minAreaRect(cnt) for cnt in contours]
    #print("tamaño rects:",len (rects))
    #print("Rectangulos contenedores:\n",rects)
    
    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap='gray')
    for rect in rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ax.add_patch(plt.Polygon(box, edgecolor='r', facecolor='none'))
    plt.show(block='TRUE')
    
    offset = percentage / 100 / 2
    enlarged_boxes = []
    for rect in rects:
        center, size, angle = rect
        width = size[0] * (1 + offset * 2)
        height = size[1] * (1 + offset * 2)
        #print("Puntos del rectangulo en objeto:\n",cv2.boxPoints(((center[0], center[1]), (width, height), angle)))
        enlarged_boxes.append(cv2.boxPoints(((center[0], center[1]), (width, height), angle)))

    # Crea una lista de imágenes con el contenido de la imagen a color en la ubicación de los rectángulos contenedores
    result_images = []
    for box in enlarged_boxes:
        rect = cv2.minAreaRect(box)
        
        #print("Rectangulos contenedores agrandados:\n",rect)
        
        center, size, angle = rect
        width = int(size[0])
        height = int(size[1])
        
        if width>height:
            width,height=height,width ## el cambio de variables mas loco del mundo
            angle=angle-90
        
        M = cv2.getRotationMatrix2D(center, angle, 1)
        
        #print("Tamaño imagen warped", (color_image.shape[1], color_image.shape[0]))
        
        rotated = cv2.warpAffine(color_image, M, (color_image.shape[1], color_image.shape[0]),borderMode=cv2.BORDER_REFLECT)
        #print("Recorte en: ",int(center[1] - height / 2),":",int(center[1] + height / 2),",", int(center[0] - width / 2),":",int(center[0] + width / 2))
        #plt.figure(figsize=(10,5))
        #plt.imshow(rotated)
        #plt.title('rotated')
        #plt.show(block='TRUE') 
        
        result_images.append(rotated[int(center[1] - height / 2):int(center[1] + height / 2), int(center[0] - width / 2):int(center[0] + width / 2)])
    result_temperatures = []
    for box in enlarged_boxes:
        rect = cv2.minAreaRect(box)
        center, size, angle = rect
        width = int(size[0])
        height = int(size[1])
        if width>height:
            width,height=height,width ## el cambio de variables mas loco del mundo
            angle=angle-90
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotatedt = cv2.warpAffine(thermal_image, M, (thermal_image.shape[1], thermal_image.shape[0]))
        result_temperatures.append(rotatedt[int(center[1] - height / 2):int(center[1] + height / 2), int(center[0] - width / 2):int(center[0] + width / 2)])
    #print(30*"*")
    #print("Tamaño de imagenes resultantes de pies")
    #print("Imagen r: ",result_images[0].shape,"Imagen l: ",result_images[1].shape)
    
    return result_images,result_temperatures

def resize_contour_boxes(binary_image, color_image, thermal_image, percentage=10):
    # Encuentra los contornos externos de la imagen binaria
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Encuentra los mínimos rectángulos contenedores de la lista de contornos
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    # Calcula el desplazamiento para aplicar el porcentaje simétricamente
    offset = percentage / 100 / 2
    # Incrementa el tamaño de los rectángulos contenedores según el porcentaje especificado
    enlarged_boxes = [(int(x - w * offset), int(y - h * offset), int(w * (1 + offset * 2)), int(h * (1 + offset * 2))) for (x, y, w, h) in boxes]
    # Crea una lista de imágenes con el contenido de la imagen a color en la ubicación de los rectángulos contenedores
    result_images = []
    for box in enlarged_boxes:
        x, y, w, h = box
        result_images.append(color_image[y:y+h, x:x+w])
    result_thermal=[]
    for box in enlarged_boxes:
        x, y, w, h = box
        result_thermal.append(thermal_image[y:y+h, x:x+w])
    return result_images,result_thermal
        
def plot_images_and_thermal(image_list,thermal_list):
    num_images = len(image_list)+len(thermal_list)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))
    for i in range(len(image_list)):
        axes[i].imshow(image_list[i])
        axes[i].axis('off')
        
    for i in range(len(thermal_list)):
        axes[len(image_list)+i].imshow(thermal_list[i], cmap='jet')
        axes[len(image_list)+i].axis('off')
        
    plt.tight_layout()
    plt.show()
    
def apply_mask(image, mask):
    # Verifica si las dimensiones de las imágenes son iguales
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Las dimensiones de las imágenes no coinciden.")

    # Crea una imagen negra del mismo tamaño que la imagen original
    masked_image = np.zeros_like(image)

    # Copia los píxeles de la imagen original donde la máscara es uno
    masked_image[mask.astype(bool)] = image[mask.astype(bool)]

    return masked_image

def rotate_image(image, direction):
    height, width = image.shape[:2]

    if width > height:
        if direction == "cw":
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif direction == "ccw":
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError("Invalid direction. Expected 'cw' or 'ccw'.")
        
        return rotated_image
    
    return image

def Find_feets(Color_Image,thermal_image,percentage=0):
    output=segment_skin(Color_Image)
    _,thresholded_image=fethearing_bin_to_color(output, Color_Image)
    binary=draw_largest_contours(thresholded_image)
    segmented_Feet,segmented_temps=resize_contour_boxes_rotated(binary, Color_Image, thermal_image, percentage=percentage)
    segmented_Feet[0]=rotate_image(segmented_Feet[0], "cw")
    segmented_temps[0]=rotate_image(segmented_temps[0], "cw")
    segmented_Feet[1]=rotate_image(segmented_Feet[1], "ccw")
    segmented_temps[1]=rotate_image(segmented_temps[1], "ccw")
    return segmented_Feet,segmented_temps

def matchbydescriptors(image_a,image_b):
    # Detectar y extraer puntos clave y descriptores en ambas imágenes
    orb = cv2.SIFT_create(5000)
    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)
    # Encontrar los puntos correspondientes entre las imágenes
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    
    matches = matcher.match(descriptors_a, descriptors_b)
    # Seleccionar los mejores N puntos correspondientes
    n_points = int(0.5*len(matches))
    matches = sorted(matches, key=lambda x: x.distance)[:n_points]
    # Extraer las coordenadas de los puntos correspondientes en ambas imágenes
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    # Calcular la matriz de homografía
    homography_matrix, _ = cv2.findHomography(points_b, points_a, cv2.RANSAC, 5.0)
    # Realizar la transformación perspectiva de la imagen B a la perspectiva de la imagen A
    image_b_transformed = cv2.warpPerspective(image_b, homography_matrix, (image_a.shape[1], image_a.shape[0]))
    # Superponer las imágenes transformadas
    alpha = 0.5
    output_image = cv2.addWeighted(image_a, 1 - alpha, image_b_transformed, alpha, 0)
    return output_image,homography_matrix

def visualize_descriptors(image_a, image_b):
    # Detectar y extraer puntos clave y descriptores en ambas imágenes
    orb = cv2.SIFT_create(5000)
    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)

    # Visualizar los puntos clave en la imagen A
    img_a_with_keypoints = cv2.drawKeypoints(image_a, keypoints_a, None)

    # Visualizar los puntos clave en la imagen B
    img_b_with_keypoints = cv2.drawKeypoints(image_b, keypoints_b, None)

    # Mostrar las imágenes con los puntos clave utilizando Matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cv2.cvtColor(img_a_with_keypoints, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Imagen A con puntos clave')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(img_b_with_keypoints, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Imagen B con puntos clave')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

def matchbydescriptors(image_a,image_b):

    # Detectar y extraer puntos clave y descriptores en ambas imágenes
    orb = cv2.ORB_create()
    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)

    # Encontrar los puntos correspondientes entre las imágenes
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)

    # Seleccionar los mejores N puntos correspondientes
    n_points = 4
    matches = sorted(matches, key=lambda x: x.distance)[:n_points]

    # Extraer las coordenadas de los puntos correspondientes en ambas imágenes
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Calcular la matriz de homografía
    homography_matrix, _ = cv2.findHomography(points_b, points_a, cv2.RANSAC, 5.0)

    # Realizar la transformación perspectiva de la imagen B a la perspectiva de la imagen A
    image_b_transformed = cv2.warpPerspective(image_b, homography_matrix, (image_a.shape[1], image_a.shape[0]))

    # Superponer las imágenes transformadas
    alpha = 0.5
    output_image = cv2.addWeighted(image_a, 1 - alpha, image_b_transformed, alpha, 0)
    return output_image,homography_matrix

def overlay_images(img1, img2, scale_factor, offset_x, offset_y, alpha):
    """
    Overlay two images on top of each other with scaling, translation, and transparency.

    Args:
        img1 (numpy.ndarray): The base image.
        img2 (numpy.ndarray): The image to overlay on top of the base image. this is a temperature image so it is a float image
        scale_factor (float): The scaling factor for img2.
        offset_x (int): The horizontal translation offset.
        offset_y (int): The vertical translation offset.
        alpha (float): The transparency level of img2. Value ranges from 0.0 to 1.0,
                       where 0.0 is fully transparent and 1.0 is fully opaque.

    Returns:
        tuple[numpy.ndarray]: A tuple containing two images:
            - The resulting image with the overlay applied.
            - A copy of the region in img1 where img2 is overlaid.

    Raises:
        ValueError: If the dimensions of img1 and img2 do not match. (not yet implemented TODO...)

    """
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    img2 = cv2.resize(img2, None, fx=scale_factor, fy=scale_factor)
    x_offset = int((img1.shape[1] - img2.shape[1]) / 2 + offset_x)
    y_offset = int((img1.shape[0] - img2.shape[0]) / 2 + offset_y)
    image_copy = np.copy(img1[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]])
    overlay = img1.copy()
    overlay[y_offset:y_offset+img2.shape[0], x_offset:x_offset+img2.shape[1]] = img2
    output = cv2.addWeighted(overlay, alpha, img1, 1 - alpha, 0)
    return(output,image_copy)

def apply_colormap(image):
    """
    Apply a false color map to a grayscale image.

    Args:
        image (numpy.ndarray): The grayscale input image.

    Returns:
        numpy.ndarray: The resulting false color image.

    """
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    colormap_image = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return colormap_image

def extract_images(flirobj, offset=[0], plot=1,transparency=0.5):
    """
    Function that matches the thermal image to the RGB image 
    
    INPUTS:
        1) flirobj: the flirimageextractor object.
        2) offset: optional variable that shifts the RGB image to match the same field of view as thermal image. 
                If not provided the offset values will be extracted from FLIR file.
        3) plot: a flag that determine if a figure of thermal and coarse cropped RGB is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) colormapthermal: a 3D numpy array of thermal image in a resolution similar of the RGB 
        2) rgbimage: a 3D numpy arary of RGB image that matches resolution and field of view of thermal image. (It has not been cropped) 
        3) temps: a 1D floating point matrix with temperatures 
        4) rgbimagecropped: a 3D rgb image cropped to correspond to the temperature matrix  
    """
    visual = flirobj.rgb_image_np
    thermalimg=flirobj.get_thermal_np()
    if len(offset) < 2:
        offsetx = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetX", "-b", flirobj.flir_img_filename])) 
        offsety = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetY", "-b", flirobj.flir_img_filename])) 
    else:
        offsetx = offset[0]
        offsety = offset[1]
    pipx2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPX2", "-b", flirobj.flir_img_filename])) # Width
    pipy2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPY2", "-b", flirobj.flir_img_filename])) # Height
    real2ir = float(subprocess.check_output([flirobj.exiftool_path, "-Real2IR", "-b", flirobj.flir_img_filename])) # conversion of RGB to Temp
    if plot:
        print(f"Image with offsetx={offsetx}, offsety={offsety}, pipx2={pipx2}, pipy2={pipy2}, real2ir={real2ir}")
    
    colormap_image = apply_colormap(thermalimg)
    scale_factor = (1/real2ir)*1080/(np.min((pipy2+1,pipx2+1))) 
    Imsalida,image_copy=overlay_images(visual, colormap_image, scale_factor, offsetx, offsety, transparency)
    rgb_height, rgb_width, _ = image_copy.shape
    temp_img_resized = cv2.resize(thermalimg, (rgb_width, rgb_height), interpolation=cv2.INTER_LANCZOS4)

    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(2,2,1)
        plt.imshow(colormap_image)
        plt.title('Thermal Image in false color')
        plt.subplot(2,2,2)
        plt.imshow(Imsalida)
        plt.title('RGB + thermal uncropped Image')
        plt.subplot(2,2,3)
        plt.imshow(temp_img_resized, cmap='jet')
        plt.title('Thermal Image resized to fit color image in jet colormap (actual temperatures)')
        plt.subplot(2,2,4)
        plt.imshow(image_copy)
        plt.title('RGB image cropped')
        plt.show(block='TRUE') 
    
    return colormap_image, Imsalida,temp_img_resized,image_copy

def save_thermal_csv(data, filename, delimiter=';'):
    """
    Function that saves the numpy array as a .csv
    
    INPUTS:
    1) data: the thermal matrix.
    2) filename: a string containing the location of the output csv file. 
    3) delimiter: use , to english and ; to spanish to save the dataset. 
    
    OUTPUTS:
    Saves a csv of the thermal image where each value is a pixel in the thermal image. 
    """
    np.savetxt(filename, data, delimiter=delimiter)# ; is the default for spanish.

def extract_coarse_image_values(flirobj, offset=[0], plot=1):
    """
    Function that creates the coarse RGB image that matches the resolution of the thermal image.
    
    INPUTS:
        1) flirobj: the flirimageextractor object.
        2) offset: optional variable that shifts the RGB image to match the same field of view as thermal image. 
                If not provided the offset values will be extracted from FLIR file. 
                Use the manual_img_registration function to determine offset.
        3) plot: a flag that determine if a figure of thermal and coarse cropped RGB is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) lowres: a 3D numpy array of RGB image that matches resolution of thermal image (It has not been cropped) 
        2) crop: a 3D numpy arary of RGB image that matches resolution and field of view of thermal image.
    """
    # Get RGB Image
    visual = flirobj.rgb_image_np
    highres_ht = visual.shape[0]
    highres_wd = visual.shape[1]
    
    # Getting Values for Offset
    if len(offset) < 2:
        offsetx = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetX", "-b", flirobj.flir_img_filename])) 
        offsety = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetY", "-b", flirobj.flir_img_filename])) 
    else:
        offsetx = offset[0]
        offsety = offset[1]
    pipx2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPX2", "-b", flirobj.flir_img_filename])) # Width
    pipy2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPY2", "-b", flirobj.flir_img_filename])) # Height
    real2ir = float(subprocess.check_output([flirobj.exiftool_path, "-Real2IR", "-b", flirobj.flir_img_filename])) # conversion of RGB to Temp
    #print(f"Image with offsetx={offsetx}, offsety={offsety}, pipx2={pipx2}, pipy2={pipy2}, real2ir={real2ir}")
    """ Example values:
    more of this parameters in 
    http://softwareservices.flir.com/BFS-U3-51S5PC/latest/Model/public/ImageFormatControl.html#OffsetX
    XoffsetX =-1 las de la mano acá tienen +1
    OffsetY=37 las de la mano acá tienen -58
    PiPX2 479
    PiPY2 639
    Real2IR 1.22885632514954
    
    """
    
    
    
    # Set up Arrays
    height_range = np.arange(0,highres_ht,real2ir).astype(int)
    width_range = np.arange(0,highres_wd,real2ir).astype(int)
    htv, wdv = np.meshgrid(height_range,width_range)
    
    # Assigning low resolution data
    lowres = np.swapaxes(visual[htv, wdv,  :], 0, 1)
    
    # Cropping low resolution data
    height_range = np.arange(-offsety+(pipy2/2),-offsety+(pipy2/2)).astype(int)
    width_range = np.arange(-offsetx+(pipx2/2),-offsetx+(pipx2/2)).astype(int)
    xv, yv = np.meshgrid(height_range,width_range)
    crop = np.swapaxes(lowres[xv, yv, :],0,1)
    
    if plot == 1:
        therm = flirobj.get_thermal_np()
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(therm, cmap='jet')
        plt.title('Thermal Image')
        plt.subplot(1,2,2)
        plt.imshow(crop)
        plt.title('RGB Cropped Image')
        plt.show(block='TRUE') 

def extract_coarse_image(flirobj, offset=[0], plot=1):
    """
    Function that creates the coarse RGB image that matches the resolution of the thermal image.
    
    INPUTS:
        1) flirobj: the flirimageextractor object.
        2) offset: optional variable that shifts the RGB image to match the same field of view as thermal image. 
                If not provided the offset values will be extracted from FLIR file. 
                Use the manual_img_registration function to determine offset.
        3) plot: a flag that determine if a figure of thermal and coarse cropped RGB is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) lowres: a 3D numpy array of RGB image that matches resolution of thermal image (It has not been cropped) 
        2) crop: a 3D numpy arary of RGB image that matches resolution and field of view of thermal image.
    """
    # Get RGB Image
    visual = flirobj.rgb_image_np
    highres_ht = visual.shape[0]
    highres_wd = visual.shape[1]
    
    # Getting Values for Offset
    if len(offset) < 2:
        offsetx = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetX", "-b", flirobj.flir_img_filename])) 
        offsety = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetY", "-b", flirobj.flir_img_filename])) 
    else:
        offsetx = offset[0]
        offsety = offset[1]
    pipx2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPX2", "-b", flirobj.flir_img_filename])) # Width
    pipy2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPY2", "-b", flirobj.flir_img_filename])) # Height
    real2ir = float(subprocess.check_output([flirobj.exiftool_path, "-Real2IR", "-b", flirobj.flir_img_filename])) # conversion of RGB to Temp
    #print(f"Image with offsetx={offsetx}, offsety={offsety}, pipx2={pipx2}, pipy2={pipy2}, real2ir={real2ir}")
    """ Example values:
    more of this parameters in 
    http://softwareservices.flir.com/BFS-U3-51S5PC/latest/Model/public/ImageFormatControl.html#OffsetX
    XoffsetX =-1 las de la mano acá tienen +1
    OffsetY=37 las de la mano acá tienen -58
    PiPX2 479
    PiPY2 639
    Real2IR 1.22885632514954
    
    """
    
    
    
    # Set up Arrays
    height_range = np.arange(0,highres_ht,real2ir).astype(int)
    width_range = np.arange(0,highres_wd,real2ir).astype(int)
    htv, wdv = np.meshgrid(height_range,width_range)
    
    # Assigning low resolution data
    lowres = np.swapaxes(visual[htv, wdv,  :], 0, 1)
    
    # Cropping low resolution data
    height_range = np.arange(-offsety+(pipy2/2),-offsety+(pipy2/2)).astype(int)
    width_range = np.arange(-offsetx+(pipx2/2),-offsetx+(pipx2/2)).astype(int)
    xv, yv = np.meshgrid(height_range,width_range)
    crop = np.swapaxes(lowres[xv, yv, :],0,1)
    
    if plot == 1:
        therm = flirobj.get_thermal_np()
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(therm, cmap='jet')
        plt.title('Thermal Image')
        plt.subplot(1,2,2)
        plt.imshow(crop)
        plt.title('RGB Cropped Image')
        plt.show(block='TRUE') 
        
    return lowres, crop

def manual_img_registration(flirobj):
    """
    Function that displays the thermal and RGB image so that similar locations 
    can be selected in both images. It is recommended that at least three tied-points
    are collected. Using the tie points the average x and y pixel offset will be determined.
    
    HOW TO:
    Left click adds points, right click removes points (necessary after a pan or zoom),
    and middle click stops point collection. 
    The keyboard can also be used to select points in case your mouse does not have one or 
    more of the buttons. The delete and backspace keys act like right clicking 
    (i.e., remove last point), the enter key terminates input and any other key 
    (not already used by the window manager) selects a point. 
    ESC will delete all points - do not use. 
    
    INPUTS:
        1) flirobj: the flirimageextractor object.
    OUTPUTS:
        1) offset: a numpy array with [x pixel offset, y pixel offset] between thermal and rgb image
        2) pts_therm: a numpy array containing the image registration points for the thermal image. 
        3) pts_rgb: a numpy array containing the coordinates of RGB image matching the thermal image.
    """
    # Getting Images
    therm = flirobj.get_thermal_np()
    rgb, junk = extract_coarse_image(flirobj)
    
    # Plot Images
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(therm, cmap='jet')
    ax1.set_title('Thermal')
    ax1.text(0,-100,'Collect points matching features between images. Select location on thermal then RGB image.')
    ax1.text(0,-75,'Right click adds a point. Left click removes most recently added point. Middle click (or enter) stops point collection.')
    ax1.text(0,-50,'Zoom/Pan add a point, but you can remove with left click. Or use back arrow to get back to original view.')
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(rgb)
    ax2.set_title('RGB')
    fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01,right=0.95)
    
    # Getting points
    pts = np.asarray(fig.ginput(-1, timeout=-1))    
    idx_therm = np.arange(0,pts.shape[0],2)
    pts_therm = pts[idx_therm,:]
    idx_rgb = np.arange(1,pts.shape[0],2)
    pts_rgb = pts[idx_rgb,:]
    
    # Getting Difference between images to determine offset
    size_therm = pts_therm.shape[0]
    size_rgb = pts_rgb.shape[0]
    offset = [0,0]
    if size_therm == size_rgb:
        pts_diff = pts_therm - pts_rgb  
        offset = np.around(np.mean(pts_diff, axis=0))
    else:
        print('Number of points do not match between images')
        
    plt.close()
    
    return offset, pts_therm, pts_rgb

def classify_rgb(img, K=3, plot=1):
    """
    This classifies an RGB image using K-Means clustering.
    Note: only 10 colors are specified, so will have plotting error with K > 10
    INPUTS:
        1) img: a 3D numpy array of rgb image
        2) K: optional, the number of K-Means Clusters
        3) plot: a flag that determine if multiple figures of classified is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) label_image: a 2D numpy array the same x an y dimensions as input rgb image, 
            but each pixel is a k-means class.
        2) result_image: a 3D numpy array the same dimensions as input rgb image, 
            but having undergone Color Quantization which is the process of 
            reducing number of colors in an image.
    """
    # Preparing RGB Image
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    
    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
    
    # Use if you want to have quantized imaged
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    
    # Labeled class image
    label_image = label.reshape((img.shape[0], img.shape[1]))

    if plot == 1:
        # Plotting Results
        coloroptions = ['b','g','r','c','m','y','k','orange','navy','gray']
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(img)
        ax1.set_title('Original Image') 
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2 = fig.add_subplot(1,2,2)
        cmap = colors.ListedColormap(coloroptions[0:K])
        ax2.imshow(label_image, cmap=cmap)
        ax2.set_title('K-Means Classes') 
        ax2.set_xticks([]) 
        ax2.set_yticks([])
        fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
        plt.show(block='TRUE')
        
        # Plotting just K-Means with label
        ticklabels = ['1','2','3','4','5','6','7','8','9','10']
        fig, ax = plt.subplots(figsize=(5,5))
        im = ax.imshow(label_image, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, shrink = 0.6, ticks=np.arange(0,K)) 
        cbar.ax.set_yticklabels(ticklabels[0:K]) 
        cbar.ax.set_ylabel('Classes')
        plt.show(block='TRUE')

    return label_image, result_image

def apply_mask_to_rgb(mask, rgbimg, plot=1):
    """
    Function that applies mask to provided RGB image and returns RGB image with 
    only pixels where mask is 1 and all other pixels are black. This function
    is useful to use BEFORE K-Means classification. 
    INPUTS:
        1) mask: a numpy binary mask that same shape as rgbimg variable. 
                0's are pixels NOT of interest and will be masked out.
                1's are pixels of interest and will be returned.
        2) rgbimg: a 3D numpy array that contains RGB image.
        3) plot: a flag that determine if a figure of masked image is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) masked_rgb: a 3D numpy array that contains RGB image with all pixels 
                designated as 0 in the mask are black. 
    """         
    masked_rgb = np.zeros((rgbimg.shape[0], rgbimg.shape[1], rgbimg.shape[2]),int)
    for d in range(0,rgbimg.shape[2]):
        masked_rgb[:,:,d] = rgbimg[:,:,d] * mask 
    
    if plot == 1:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(rgbimg)
        plt.title('Original Image')
        plt.subplot(1,2,2)
        plt.imshow(masked_rgb)
        plt.title('Masked Image')
        plt.show(block='TRUE')
        
    return masked_rgb

def create_class_mask(classimg, classinterest, plot=1):
    """
    This function creates a mask that turns all K-Means classes NOT of interest 
    as 0 and all classes of interest to 1. This can be used to extract temperatures
    only for classes of interest. 
    INPUTS:
        1) classinterest: a array containing the class or classes of interest. 
                All other classes will be masked out
        2) classimg: the K-Means class image which is (2D) numpy array
        3) plot: a flag that determine if a figure of masked K-Means class image is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) mask: a 2D numpy binary mask that same shape as the first two dimensions of rgbimg variable. 
                0's are pixels NOT of interest and will be masked out.
                1's are pixels of interest and will be returned.
    """
    mask = np.zeros((classimg.shape[0], classimg.shape[1]))
    if isinstance(classinterest,int):
        endrange = 1
    else:
        endrange = len(classinterest)
    
    for c in range(0,endrange):
        idx_x, idx_y = np.where(classimg == c)
        mask[idx_x, idx_y] = 1
    
    if plot == 1:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(classimg)
        plt.title('Original K-Means Classes')
        plt.subplot(1,2,2)
        plt.imshow(mask, cmap='gray')
        plt.title('Masked Classes')
        plt.show(block='TRUE')
    
    return mask

def extract_temp_for_class(flirobj, mask, emiss=[0], plot=1):
    """
    This function creates a numpy array thermal image that ONLY contains pixels for class
    of interest with all other pixels set to 0. This is for a SINGLE image
    INPUTS:
        1) flirobj: the flirimageextractor object
        2) mask: a binary mask with class pixels set as 1. 
        3) emiss: OPTIONAL, a 2D numpy array with each pixel containing correct emissivity
                If provided the temperature will be corrected for emissivity
        4) plot: a flag that determine if a figure of tempeature image is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) therm_masked: a 2D numpy array of temperature values for a class of interest
    """       
    if len(emiss) == 1:
        therm = flirobj.get_thermal_np()
    else:
        therm = correct_temp_emiss(flirobj, emiss, plot=0)
        
    therm_masked = np.ma.masked_where(mask != 1, therm)
    
    if plot == 1:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(therm, cmap='jet')
        plt.subplot(1,2,2)
        plt.imshow(therm_masked, cmap='jet')
        plt.show(block='TRUE')
        
    return therm_masked

def batch_extract_temp_for_class(dirLoc, mask, emiss=[0], exiftoolpath=''):
    """
    This function creates a 3D numpy array thermal image that ONLY contains pixels for class
    of interest with all other pixels set to 0. This is for a directory of images.
    INPUTS:
        1) dirLoc: a string containing the directory of FLIR images.
        2) mask: a binary mask with class pixels set as 1. 
        3) emiss: a 2D numpy array with each pixel containing correct emissivity
        4) exiftoolpath = OPTIONAL a string containing the location of exiftools.
                Only use if the exiftool path is different than the python path.
    OUTPUTS:
        1) all_temp: a 3D numpy array of temperature values for a class of interest
                The first two dimensions will match the dimensions of a single temperature image.
                The third dimension size will match the number of files in directory.
    """  
    filelist = glob.glob(dirLoc + '*')
    #print('Found ' + str(len(filelist)) + ' files.')
    all_temp = np.ma.empty((mask.shape[0], mask.shape[1], len(filelist)))
    
    for f in range(0,len(filelist)):
        # Get individual file
        if not exiftoolpath:
            flir = flirimageextractor.FlirImageExtractor()
        else:
            flir = flirimageextractor.FlirImageExtractor(exiftool_path=exiftoolpath)

        flir.process_image(filelist[f], RGB=True)
        
        if len(emiss) == 1:
            all_temp[:,:,f] = extract_temp_for_class(flir, mask, plot=0)
        else:
            all_temp[:,:,f] = extract_temp_for_class(flir, mask, emiss, plot=0)
        
    return all_temp

def plot_temp_timeseries(temp):
    """
    Function that plots the mean, min, and max temperature for a temperature timeseries.
    INPUTS:
        1) temp: a 3D numpy array of temperature values for a class of interest
                The first two dimensions will match the dimensions of a single temperature image.
                The third dimension size will match the number of files in directory. 
    OUTPUTS:
        figure of mean, min, and maximum temperature across timeseries
    """
    # Setting up Variables
    minmaxtemp = np.zeros((2,temp.shape[2]))
    meantemp = np.zeros(temp.shape[2])
    
    # Loop through time steps
    for d in range(0, temp.shape[2]):
        minmaxtemp[0,d] = np.nanmin(temp[:,:,d])
        meantemp[d] = np.nanmean(temp[:,:,d])
        minmaxtemp[1,d] = np.nanmax(temp[:,:,d])
    
    difftemp = abs(minmaxtemp - meantemp)
    plt.figure(figsize=(10,7))
    plt.errorbar(np.arange(1, temp.shape[2]+1), meantemp, yerr=difftemp, linewidth=2, color='black')
    plt.gca().yaxis.grid(True)
    plt.title('Temperature through Timeseries')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (Celsius)')
    plt.show(block='true')
    
def develop_correct_emissivity(class_img):
    """
    The thermal camera has an assume emissivity of 0.95, but many materials do 
    not have that emissivity which changes the temperature retrieved. This code
    assigned the appropriate emissivity value for a pixel (user provided) 
    using the K-Means classes.
    INPUTS:
        1) class_img: the K-Means class image
    OUTPUTS:
        1) emiss_img: a numpy array with same dimensions as K-Means class image,
                but every pixel has an emissivity value.
    """
    K = len(np.unique(class_img))
    
    # Plotting just K-Means with label
    coloroptions = ['b','g','r','c','m','y','k','orange','navy','gray']
    ticklabels = ['1','2','3','4','5','6','7','8','9','10']
    fig, ax = plt.subplots(figsize=(5,5))
    cmap = colors.ListedColormap(coloroptions[0:K])
    im = ax.imshow(class_img, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, shrink = 0.6, ticks=np.arange(0,K)) 
    cbar.ax.set_yticklabels(ticklabels[0:K]) 
    cbar.ax.set_ylabel('Classes')
    plt.show(block='true')
    
    #print('Input the emissivity for each class. If unknown put 0.95')
    emiss = np.zeros((K))
    for c in range(0,K):
        strout = 'Emissivity for Class ' + str(c+1) + ': '
        emiss[c] = input(strout)
        
    emiss_img = np.zeros((class_img.shape[0], class_img.shape[1]))    
    for e in range(0, K):
        idx_x, idx_y = np.where(class_img == e)
        emiss_img[idx_x, idx_y] = emiss[e]
        
    return emiss_img

def correct_temp_emiss(flirobj, emiss, plot=1):
    """
    The thermal camera has an assume emissivity of 0.95, but many materials do 
    not have that emissivity which changes the temperature retrieved. This function 
    takes in the user provided emissivity array for each pixel and corrects the
    temperature values.
    This uses the stephan boltzman equation. 
    INPUTS:
        1) flirobj: a flirimageextractor object
        2) emiss: a 2D numpy array with each pixel containing correct emissivity
                Using these values the temperature will be corrected for emissivity
    OUTPUTS:
        1) corrected_temp: a 2D numpy array with corrected temperature values
    """    
    therm = flirobj.get_thermal_np()
    
    sbconstant = 0.00000005670374419 
    
    # Get total flux using the assumed emissivity of 0.95
    totalflux = 0.95 * sbconstant * np.power(therm, 4)
    
    # Solving for Temperature given new emissivities
    value = totalflux/(emiss*sbconstant)
    corrected_temp = np.power(value,1/4)
    
    if plot == 1:
        plt.figure(figsize=(5,5))
        plt.imshow(corrected_temp, cmap='jet')
        plt.colorbar()
        plt.show(block='true')
        
    return corrected_temp




