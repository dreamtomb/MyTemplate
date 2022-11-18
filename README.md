# MyTemplate项目说明
## 一、项目目的
本项目是自用的深度学习框架模板，用于快速编写程序进行实验。
## 二、项目结构
|---README.md
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
## 三、项目结构说明
- README.md文件是项目说明
- augmentation文件夹包含了常用的数据增强操作
- checkpoints文件夹用于存放不同迭代的模型参数
- config文件夹存放了配置文件，用于指定与实验相关的所有超参数、设置
- dataset文件夹用于读取数据集
- image_res文件夹用于存放测试中保存的图像结果
- log文件夹用于记录实验输出的日志
- model文件夹用于搭建模型
- pretrain_model文件夹用于存放预训练模型
- record文件夹用于保存一次实验的完整记录：日志、权重、可视化、配置、备份
- solver文件夹用于设置loss和优化器
- summary文件夹用于总结不同设置下模型的表现
- trainer文件夹用于编写训练和测试代码
- utils文件夹包含了常用的工具函数
## 四、需要调整的内容
- config文件调整超参数
- main、train、test文件根据不同的模型进行修改