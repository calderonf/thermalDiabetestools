# ThermalDiabetesTools

**ThermalDiabetesTools** is a software tool designed to analyze a temporal series of thermal images captured using the FLIR One Pro camera. The tool aims to predict the likelihood of a person developing diabetes based on the analysis of temperature progression at various points on their feet.

## Features

- A library to capture and process thermal RGB and perform image registration of images captured with flir one pro cameras
- Import a temporal series of thermal images captured with the FLIR One Pro camera.
- Analyze the temperature progression over time at different points on the feet or at the complete feet.
- Perform statistical analysis and modeling to predict the likelihood of diabetes.
- Generate informative visualizations and reports to aid in decision-making.

## Installation

0. Download and decompress exiftool ( https://exiftool.org/ ) and then paste it into windows folder ("C:\\Windows\\exiftool.exe")

1. Clone the repository:

   ```bash
   git clone https://github.com/calderonf/ThermalDiabetesTools.git
   ```

2. Install anaconda Python https://www.anaconda.com/

3. Install the required dependencies:

   ```bash
   conda env create -f environment.yml
   ```

### to install sdk of flir (to this moment is optional)

Download it from web page, https://flir.custhelp.com/app/account/fl_download_software and then select SDK and FLIR science file sdk i download and tested the windows one. 

decompress and install the .exe in my case was "C:\Program Files\FLIR Systems\sdks" the installation folder

in "file:///C:/Program%20Files/FLIR%20Systems/sdks/file/python/doc/index.html" there is a documentation of the python sdk. 

there was no whl in my sdk i have to install it:
you will need a visual studio (https://visualstudio.microsoft.com/es/thank-you-downloading-visual-studio/?sku=Community&rel=15) for the version 3.8 in windows the compiler was vs2017 install it with C++ support
   ```bash
   cd "C:\Program Files\FLIR Systems\sdks\file\python"
   conda activate thermal
   pip install setuptools cython wheel
   python setup.py install --shadow-dir C:\tempflir
   ```
   now find in this temporal directory for the *.whl file in my case python 3.8 and AMD64.
   an install it using 
   ```bash
   pip install .\FileSDK-4.1.0-cp38-cp38-win_amd64.whl
   ```


## Usage

1. Launch the `ThermalDiabetesTools` application.

2. Import the temporal series of thermal images captured with the FLIR One Pro camera.

3. Perform the analysis to analyze the temperature progression at various points on the feet.

4. Utilize the statistical analysis and modeling capabilities to predict the likelihood of diabetes.

5. Generate visualizations and reports to assist in decision-making and further analysis.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue on the GitHub repository. If you would like to contribute code, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).?????

## Acknowledgements

- The `ThermalDiabetesTools` software was inspired by the work of researchers in the field of thermal imaging and diabetes prediction.
- the exiftool project https://github.com/exiftool/exiftool , https://exiftool.org/ , and exiftoolgui https://exiftool.org/forum/index.php?topic=2750.0
- FLIR_thermal_tools project https://github.com/susanmeerdink/FLIR_thermal_tools
- flir image extractor project (https://pypi.org/project/flirimageextractor/) https://github.com/nationaldronesau/FlirImageExtractor
- read_thermal_temperature project https://github.com/ManishSahu53/read_thermal_temperature
- partial acknowledmen to Flir and teledine corporation to the creation of this software, your lack and difficulti of acces to the information of your cameras was vital for the creation of this library (https://flir.custhelp.com/app/account/fl_download_software)


## Contact

For any inquiries or further information, please contact our team at calderonf@javeriana.edu.co

## Disclaimer

The `ThermalDiabetesTools` software is not intended to replace professional medical advice or diagnosis. It is a tool meant to assist in the analysis and prediction of diabetes based on thermal imaging data. Always consult with a healthcare professional for accurate diagnosis and treatment.
