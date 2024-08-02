import os
import re
import csv

def contains_keywords(file_path, keywords):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        for keyword in keywords:
            if keyword in content:
                return True
    return False

def extract_test_cases(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        test_cases = re.findall(r'Torch\w*TestCase', content)
        return test_cases

def collect_test_cases(directory, keywords):
    test_cases_dict = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                file_path = os.path.join(root, file)
                if contains_keywords(file_path, keywords):
                    test_cases = extract_test_cases(file_path)
                    if test_cases:
                        test_cases_dict[file_path] = test_cases
    return test_cases_dict

def write_test_cases_to_csv(test_cases_dict, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['File Path', 'Test Cases'])
        for file_path, test_cases in test_cases_dict.items():
            csvwriter.writerow([file_path, ', '.join(test_cases)])

# 关键字列表
modules = [
    "torch",
    "torch.random", # no testcases
    "torch.utils.cpp_extension", # skip
    "torch.xpu", # skip
    "torch.mps", # skip
    "torch.jit",
    "torch.onnx", # no testcases
    "torch.cuda", # skip
    "torch.linalg",
    "torch.amp", # no testcases
    "torch.distributed", # no testcases
    "torch.Tensor",
    "torch.fx", # skip
    "torch.hub", # skip
    "torch.masked", # skip
    "torch.utils.tensorboard", # skip
]

# 示例使用
directory_path = "src/testcase/torch/"
output_csv_file = "test_cases.csv"
test_cases_dict = collect_test_cases(directory_path, modules)
write_test_cases_to_csv(test_cases_dict, output_csv_file)
print(f"已将测试用例写入 {output_csv_file}")
