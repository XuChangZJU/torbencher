# from .broadcast_shapes import TorchBroadcastshapesTestCase
# from .stack import TorchStackTestCase
# from .normal import TorchNormalTestCase
# from .lobpcg import TorchLobpcgTestCase
# from .set_num_interop_threads import TorchSetnuminteropthreadsTestCase
# from .export import TorchExportTestCase
# from .set_default_device import TorchSetdefaultdeviceTestCase
# from ._assert import TorchAssertTestCase
# from .tensor_split import TorchTensorsplitTestCase
# from .sym_max import TorchSymmaxTestCase
# from .quantize_per_channel import TorchQuantizeperchannelTestCase
# from .broadcast_to import TorchBroadcasttoTestCase
# from .sparse_csc_tensor import TorchSparsecsctensorTestCase
# from .vstack import TorchVstackTestCase
# from .slice_scatter import TorchSlicescatterTestCase
# from .Generator import TorchGeneratorTestCase
# from .istft import TorchIstftTestCase
# from .column_stack import TorchColumnstackTestCase
# from .set_flush_denormal import TorchSetflushdenormalTestCase
# from .repeat_interleave import TorchRepeatinterleaveTestCase
# from .sym_min import TorchSymminTestCase
# from .set_rng_state import TorchSetrngstateTestCase
# from .frombuffer import TorchFrombufferTestCase
# from .Gradient import (TorchGradientTestCase)
# from .get_float32_matmul_precision import TorchGetfloat32matmulprecisionTestCase
# from .dsplit import TorchDsplitTestCase
# from .index_reduce import TorchIndexreduceTestCase
# from .reshape import TorchReshapeTestCase
# from .sym_ite import TorchSymiteTestCase

import importlib
import logging
import os

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
