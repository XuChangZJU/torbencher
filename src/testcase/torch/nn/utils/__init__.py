# from .fuse_linear_bn_eval import TorchNnUtilsFuselinearbnevalTestCase
# from .weight_norm import TorchNnUtilsWeightnormTestCase
# from .convert_conv3d_weight_memory_format import TorchNnUtilsConvertconv3dweightmemoryformatTestCase
# from .clip_grad_norm_ import TorchNnUtilsClipgradnormTestCase
# from .spectral_norm import TorchNnUtilsSpectralnormTestCase
# from .parameters_to_vector import TorchNnUtilsParameterstovectorTestCase
# from .fuse_linear_bn_weights import TorchNnUtilsFuselinearbnweightsTestCase
# from .convert_conv2d_weight_memory_format import TorchNnUtilsConvertconv2dweightmemoryformatTestCase
# from .clip_grad_value_ import TorchNnUtilsClipgradvalueTestCase
# from .skip_init import TorchNnUtilsSkipinitTestCase
# from .vector_to_parameters import TorchNnUtilsVectortoparametersTestCase
# from .fuse_conv_bn_weights import TorchNnUtilsFuseconvbnweightsTestCase
# from .remove_weight_norm import TorchNnUtilsRemoveweightnormTestCase
# from .clip_grad_norm import TorchNnUtilsClipgradnormTestCase
# from .remove_spectral_norm import TorchNnUtilsRemovespectralnormTestCase
# from .fuse_conv_bn_eval import TorchNnUtilsFuseconvbnevalTestCase

import os
import importlib
import logging

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase

# 设置日志配置
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_directory: str = os.path.dirname(os.path.abspath(__file__))
script_files: list = [f for f in os.listdir(current_directory) if f.endswith('.py') and f != '__init__.py']

for script_file in script_files:
    module_name: str = script_file[:-3]  # Remove the .py extension
    try:
        module = importlib.import_module(f'.{module_name}', package=__package__)
        # logger.debug(f"Successfully imported module {module_name}")
    except Exception as e:
        logger.debug(f"Failed to import module {module_name}: {e}")
        continue

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and issubclass(attribute, TorBencherTestCaseBase):
            globals()[attribute_name] = attribute
