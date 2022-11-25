# MyTemplate项目说明
## 一、项目说明
- 本项目是自用的深度学习框架模板，用于快速编写程序进行实验。
- 本项目使用black+flake8+isort进行代码格式化。
- 本项目使用autodocstring+googledoc格式进行函数注释。
## 二、项目结构
```
+---.vscode
+---augmentation
+---checkpoints
+---config
+---dataset
+---image_res
+---log
+---model
+---pretrain_model
+---record
+---solver
+---summary
+---trainer
+---utils
|---.gitignore
|---del.sh
|---main.py
|---README.md
|---wait_gpu.py
```
## 三、项目结构说明
- .vscode文件夹设置了python的extraPath，防止import报错
- augmentation文件夹包含了常用的数据增强操作
- checkpoints文件夹用于存放不同实验的模型参数，每次实验以当前时间为文件夹名
- config文件夹存放了配置文件，用于指定与实验相关的所有超参数、设置
- dataset文件夹用于读取数据集（病灶数据集，image和mask均为单通道）
- image_res文件夹用于存放测试中保存的图像结果，每次实验以当前时间为文件夹名
- log文件夹用于记录实验输出的日志（txt）以及tensorboard文件（events），每次实验以当前时间为文件夹名
- model文件夹用于搭建模型，现有PFSNet+resnet50以及SINet-V2+res2net50两种
- pretrain_model文件夹用于存放预训练模型
- record文件夹用于保存一次实验的所有代码，防止summary中的实验记录与代码版本不一致
- solver文件夹用于设置loss
- summary文件夹用于总结不同设置下模型的表现，snapshot自动将实验结果、超参数以及模型和日志的保存路径等填入excel文件
- trainer文件夹用于编写训练和测试代码
- utils文件夹包含了常用的工具函数和评价指标
- .gitignore文件指定了不需要git跟踪的文件（pth等）
- del.sh文件包含了三行命令，isort整理import顺序，删除pycache缓存，删除某次实验的所有文件夹
- main.py文件是运行的主函数文件
- README.md文件是项目说明
- wait_gpu.py文件是gpu排队文件，每分钟查询一次显卡空余，有空余则运行main.py