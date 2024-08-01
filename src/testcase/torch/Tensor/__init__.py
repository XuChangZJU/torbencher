# from .index_reduce import TorchTensorIndexreduceTestCase
# from .expand import TorchTensorExpandTestCase
# from .to import TorchTensorToTestCase
# from .index_put import TorchTensorIndexputTestCase
# from .is_leaf import TorchTensorIsleafTestCase
# from .diagonal_scatter import TorchTensorDiagonalscatterTestCase
# from .addr_ import TorchTensorAddrTestCase
# from .dim_order import TorchTensorDimorderTestCase
# from .expand_as import TorchTensorExpandasTestCase
# from .istft import TorchTensorIstftTestCase
# from .record_stream import TorchTensorRecordstreamTestCase
# from .renorm_ import TorchTensorRenormTestCase
# from .inner import TorchTensorInnerTestCase
# from .smm import TorchTensorSmmTestCase
# from .dsplit import TorchTensorDsplitTestCase
# from .index_put_ import TorchTensorIndexputTestCase
# from .cross import TorchTensorCrossTestCase
# from .hsplit import TorchTensorHsplitTestCase
# from .broadcast_to import TorchTensorBroadcasttoTestCase
# from .module_load import TorchTensorModuleloadTestCase
# from .index_copy import TorchTensorIndexcopyTestCase
# from .ormqr import TorchTensorOrmqrTestCase
# from .select_scatter import TorchTensorSelectscatterTestCase
# from .is_pinned import TorchTensorIspinnedTestCase
# from .lu_solve import TorchTensorLusolveTestCase
# from .reshape import TorchTensorReshapeTestCase
# from .is_cuda import TorchTensorIscudaTestCase
# from .slice_scatter import TorchTensorSlicescatterTestCase
# from .pin_memory import TorchTensorPinmemoryTestCase
# from .stft import TorchTensorStftTestCase
# from .as_strided import TorchTensorAsstridedTestCase
# from .cuda import TorchTensorCudaTestCase
# from .sspaddmm import TorchTensorSspaddmmTestCase
# from .tensor_split import TorchTensorTensorsplitTestCase
# from .element_size import TorchTensorElementsizeTestCase
# from .vsplit import TorchTensorVsplitTestCase
# from .float_power_ import TorchTensorFloatpowerTestCase
# from .multinomial import TorchTensorMultinomialTestCase

import os
import importlib
import logging
from inspect import isclass

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
    try:
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isclass(attribute) and issubclass(attribute, TorBencherTestCaseBase)\
                    and attribute is not TorBencherTestCaseBase:
                globals()[attribute_name] = attribute
    except Exception as e:
        raise ValueError(f"The testcase that cause error is {attribute_name}") from e

