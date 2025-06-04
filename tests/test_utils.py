import importlib.util
import sys
import types
import unittest

def load_derivative_utils():
    # Stub external dependencies not available in the test environment
    for name in ['flirimageextractor', 'flir_image_extractor', 'cv2', 'numpy', 'SimpleITK', 'Utils']:
        sys.modules.setdefault(name, types.ModuleType(name))

    matplotlib = types.ModuleType('matplotlib')
    matplotlib.use = lambda *a, **k: None
    pyplot = types.ModuleType('matplotlib.pyplot')
    colors = types.ModuleType('matplotlib.colors')
    matplotlib.pyplot = pyplot
    matplotlib.colors = colors
    sys.modules.setdefault('matplotlib', matplotlib)
    sys.modules.setdefault('matplotlib.pyplot', pyplot)
    sys.modules.setdefault('matplotlib.colors', colors)

    sys.modules.setdefault('registration_simpleitk', types.ModuleType('registration_simpleitk'))

    scipy = types.ModuleType('scipy')
    interpolate = types.ModuleType('scipy.interpolate')
    interpolate.interp1d = lambda *a, **k: None
    scipy.interpolate = interpolate
    sys.modules.setdefault('scipy', scipy)
    sys.modules.setdefault('scipy.interpolate', interpolate)

    spec = importlib.util.spec_from_file_location('derivative_utils', 'derivative_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestObtenerNombresHdf5(unittest.TestCase):
    def test_expected_names(self):
        du = load_derivative_utils()
        result = du.obtener_nombres_hdf5('flir_20230912T120000.jpg')
        expected = (
            'tr_right_20230912T120000.hdf5',
            'tr_left_20230912T120000.hdf5',
            'temp_right_20230912T120000.csv',
            'temp_left_20230912T120000.csv',
            'mask_right_20230912T120000.png',
            'mask_left_20230912T120000.png',
            'color_right_20230912T120000.png',
            'color_left_20230912T120000.png',
            'Raw_mask20230912T120000.png',
            'Raw_RGB20230912T120000.png',
            'Raw_temp20230912T120000.csv',
        )
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
