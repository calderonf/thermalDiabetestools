# ThermalDiabetesTools

![ThermalDiabetesTools](https://github.com/calderonf/thermalDiabetestools/blob/main/logo_thermal_%20diabetes_tools_cuadrado.png?raw=true)

**ThermalDiabetesTools** is a software tool designed to analyze a temporal series of thermal images captured using the FLIR One Pro camera. The primary aim is to predict the likelihood of a person developing diabetes based on the analysis of temperature progression at various points on their feet.

## Features

- Capture and process synchronized thermal and RGB frames from FLIR One Pro cameras.
- Import and manage temporal series of thermal images.
- Track temperature progression at custom points or across the entire foot.
- Apply registration and filtering algorithms to improve image alignment.
- Perform statistical analysis and modeling to predict the likelihood of diabetes.
- Generate informative visualizations and reports to assist in decision‑making.

## Installation

0. Download and decompress ExifTool (https://exiftool.org/). Then place it in the Windows folder ("C:\\Windows\\exiftool.exe"). Remove the `-k` from the filename.

1. Clone the repository:

   ```bash
   git clone https://github.com/calderonf/ThermalDiabetesTools.git
   cd ThermalDiabetesTools
   ```

2. Install Anaconda Python from https://www.anaconda.com/.

3. Install the required dependencies:

   ```bash
   conda env create -f environment.yml
   ```

   To activate this environment, use:

   ```bash
   conda activate thermal
   ```

   To deactivate the active environment, use:

   ```bash
   conda deactivate
   ```

### Installing the FLIR SDK (optional)

Download the SDK from the webpage (https://flir.custhelp.com/app/account/fl_download_software), and select the FLIR Science File SDK. Download and test the Windows version.

Decompress and install the `.exe` file, typically located in "C:\Program Files\FLIR Systems\sdks".

For Python SDK documentation, refer to "file:///C:/Program%20Files/FLIR%20Systems/sdks/file/python/doc/index.html".

If there is no `.whl` file in the SDK, you will need to install it manually. Visual Studio is required (https://visualstudio.microsoft.com/es/thank-you-downloading-visual-studio/?sku=Community&rel=15). For version 3.8 on Windows, use the VS2017 compiler and install it with C++ support.

   ```bash
   cd "C:\Program Files\FLIR Systems\sdks\file\python"
   conda activate thermal
   pip install setuptools cython wheel
   python setup.py install --shadow-dir C:\tempflir
   ```

   Next, locate the `.whl` file in the temporary directory (e.g., for Python 3.8 and AMD64) and install it:

   ```bash

   pip install .\FileSDK-4.1.0-cp38-cp38-win_amd64.whl
   ```

### Running Tests

The repository includes a small test suite located in the `tests/` folder.
After installing the dependencies you can execute the tests with `pytest`:

```bash
pytest
```

## Usage

1. Launch the `ThermalDiabetesTools` application.

2. Import the temporal series of thermal images captured with the FLIR One Pro camera.

3. Perform the analysis to track the temperature progression at various points on the feet.

4. Utilize the statistical analysis and modeling capabilities to predict the likelihood of diabetes.

5. Generate visualizations and reports to assist in decision-making and further analysis.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue on the GitHub repository. If you would like to contribute code, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- The `ThermalDiabetesTools` software was inspired by the work of researchers in the field of thermal imaging and diabetes prediction.
- ExifTool project: https://github.com/exiftool/exiftool, https://exiftool.org/, and ExifToolGUI: https://exiftool.org/forum/index.php?topic=2750.0
- FLIR Thermal Tools project: https://github.com/susanmeerdink/FLIR_thermal_tools
- FLIR Image Extractor project: (https://pypi.org/project/flirimageextractor/) https://github.com/nationaldronesau/FlirImageExtractor
- Read Thermal Temperature project: https://github.com/ManishSahu53/read_thermal_temperature
- Partial acknowledgement to FLIR and Teledyne Corporation for their indirect role in the creation of this software; the lack of and difficulty in accessing information about their cameras was vital for the development of this library (https://flir.custhelp.com/app/account/fl_download_software).

## How to cite

If you use **ThermalDiabetesTools** in your research, please cite the following publication:

Calderon, F.C., Zequera-Díaz, M., Gerlein, E.A., Naemi, R. (2025). *The Development of A Comprehensive Library for Thermal Image Analysis of Diabetic Feet: ThermalDiabetesTools*. In: Ballarin, V.L., Martinez-Licona, F., Pérez-Buitrago, S.M., Ibarra-Ramírez, E.A., Berriere, L.R. (eds) **1st IFMBE Latin American Conference on Digital Health**. CLASD 2024. IFMBE Proceedings, vol 119. Springer, Cham. [https://doi.org/10.1007/978-3-031-88064-3_10](https://doi.org/10.1007/978-3-031-88064-3_10)

## Contact

For any inquiries or further information, please contact our team at [calderonf@javeriana.edu.co](mailto:calderonf@javeriana.edu.co).

## Disclaimer

The `ThermalDiabetesTools` software is not intended to replace professional medical advice or diagnosis. It is a tool designed to assist in the analysis and prediction of diabetes based on thermal imaging data. Always consult with a healthcare professional for accurate diagnosis and treatment.

