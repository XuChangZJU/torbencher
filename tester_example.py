
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.singleTester import SingleTester
from src.testcase.torch.add import TorchAddTestCase

tester = SingleTester()



pass_result_list = []
for i in range(100):
    pass_result = tester.run(TorchAddTestCase, device="cpu", seed=i)
    pass_result_list.append(pass_result)

# print(pass_result_list)

if False in pass_result_list:
    print("Test failed")
else:
    print("Test passed")

