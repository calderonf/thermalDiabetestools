import SimpleITK as sitk
import numpy as np
import sys

# tomado de https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod4_docs.html

# also from: https://github.dev/DlutMedimgGroup/MedImg_Py_Library/blob/8fa35282f05ccdc73e08866e9e04db7600f7026a/Medimgpy/itk2DImageRegistration.py#L70#L72

# https://simpleitk.readthedocs.io/en/master/registrationOverview.html


def cv2_grid_sampling(fixed, moving, displacement, is_binary=False):
    # Convert the numpy arrays to SimpleITK images
    fixed_sitk = sitk.GetImageFromArray(fixed)
    moving_sitk = sitk.GetImageFromArray(moving)
    out_sitk= sitk_grid_sampling(fixed_sitk,moving_sitk, displacement,is_binary=is_binary)
    return sitk.GetArrayFromImage(out_sitk)

def sitk_grid_sampling(fixed,moving, displacement,is_binary=False):
    """_summary_

    Args:
        fixed (sitk image): reference image
        moving (sitk image): image to resample
        displacement (_type_): transformation
        is_binary (bool, optional): dice si es binaria o termica. Defaults to False.

    Returns:
        _type_: image transformated
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    interpolator = sitk.sitkNearestNeighbor if is_binary else sitk.sitkLinear
    resampler.SetInterpolator(interpolator)
    #valor por defecto del pixel
    defaultTemp= 0 if is_binary else 18
    resampler.SetDefaultPixelValue(defaultTemp)
    resampler.SetTransform(displacement)
    out = resampler.Execute(moving)
    return out

def sitk_image_to_opencv(image, force_8bits=True):
    image_np = sitk.GetArrayFromImage(image)
    return image_np.astype(np.uint8)

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

def register_images(fixed_image_np, moving_image_np, output_transform_file,
                    number_of_bins=24, sampling_percentage=0.10):
    
        # Convertir las imágenes a escala de grises
    image_a_gray = np.dot(fixed_image_np, [0.2989, 0.5870, 0.1140]).astype(np.float32)
    image_b_gray = np.dot(moving_image_np, [0.2989, 0.5870, 0.1140]).astype(np.float32)

    fixed = sitk.GetImageFromArray(image_a_gray.astype(np.float32))
    moving = sitk.GetImageFromArray(image_b_gray.astype(np.float32))

    # Convertir imágenes a escala de grises si son multicanal
    if fixed.GetNumberOfComponentsPerPixel() > 1:
        fixed = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkFloat32)
    if moving.GetNumberOfComponentsPerPixel() > 1:
        moving = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)

    # Verificar el tamaño de las imágenes
    
    if fixed.GetSize()[0] < 4 or moving.GetSize()[0] < 4:
        print(fixed.GetSize())
        print(moving.GetSize())
        raise ValueError("Las imágenes en la dirección 0 deben tener al menos 4 píxeles")

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(number_of_bins)
    R.SetMetricSamplingPercentage(sampling_percentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(0.5, 0.001, 400)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")

    sitk.WriteTransform(outTx, output_transform_file)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    fixed_np = sitk.GetArrayFromImage(fixed)
    moving_np = sitk.GetArrayFromImage(out)



    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)


    return fixed_np, moving_np, sitk_image_to_opencv(cimg)

def register_images_Similarity2DTransform(fixed_image, moving_image, output_transform_file, numberOfBins=24, samplingPercentage=0.01):
    
        
        # Convertir las imágenes a escala de grises
    image_a_gray = np.dot(fixed_image, [0.2989, 0.5870, 0.1140]).astype(np.float32)
    image_b_gray = np.dot(moving_image, [0.2989, 0.5870, 0.1140]).astype(np.float32)

    fixed_image = sitk.GetImageFromArray(image_a_gray.astype(np.float32))
    moving_image = sitk.GetImageFromArray(image_b_gray.astype(np.float32))

    # Convertir imágenes a escala de grises si son multicanal
    if fixed_image.GetNumberOfComponentsPerPixel() > 1:
        fixed_image = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkFloat32)
    if moving_image.GetNumberOfComponentsPerPixel() > 1:
        moving_image = sitk.Cast(sitk.RescaleIntensity(moving_image), sitk.sitkFloat32)

    # Verificar el tamaño de las imágenes
    
    if fixed_image.GetSize()[0] < 4 or moving_image.GetSize()[0] < 4:
        print(fixed_image.GetSize())
        print(moving_image.GetSize())
        raise ValueError("Las imágenes en la dirección 0 deben tener al menos 4 píxeles")

    
    
    R = sitk.ImageRegistrationMethod()

    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(0.1, 0.0005, 800)
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Similarity2DTransform(),sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initial_transform)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
    
    R.SetOptimizerScalesFromPhysicalShift()

    
    outTx = R.Execute(fixed_image, moving_image)

    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f"Iteration: {R.GetOptimizerIteration()}")
    print(f"Metric value: {R.GetMetricValue()}")

    sitk.WriteTransform(outTx, output_transform_file)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving_image)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed_image), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)

    return {
        "fixed": fixed_image,
        "moving": moving_image,
        "composition": cimg,
        "Transform":outTx
    }   
