# Wrapper

一个对pytorch包进行封装，对调用进行记录的工具

## 使用

先`git clone`项目到本地

将需要运行的程序改写为如下代码：

```python
from src.Wrapper import Wrapper

wrapper = Wrapper({
    "out_dir": "./result",
    "format": "csv",
    "file_max_size": "512MB",
    "file_name_spec": "timestamp"
})

def my_code():
    """
    此处写需要运行的程序
    """

wrapper.start(my_code)
```

然后执行程序，对pytorch的调用序列就会被顺序输出到`out_dir`目录下的文件中

## 配置参数

- `out_dir`: 输出目录（必需）
- `format`: 输出格式目前支持`csv`和`json`，默认为`csv`
- `file_max_size`：输出文件的最大值，超过会自动输出到下一个文件中
- `file_name_spec`：输出文件名规范，`timestamp`时间戳，`datetime`以时间值（字符串），`serial`以6位正整数（如果文件夹下原来就有输出文件，会接着最大文件名命名）

## 输出格式：

csv文件的每行代表一次调用，格式为：

```
接口名,调用次数,调用时间戳,耗费时间(ms),scale字符串(json化)
```

其中scale代表本次调用的参数的scale被JSON.stringify后的字符串，scale的标定规则与bencher保持一致