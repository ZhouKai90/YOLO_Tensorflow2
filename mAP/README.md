这部分是使用的https://github.com/Cartucho/mAP绘制mAP

并且参考https://github.com/bubbliiiing/count-mAP-txt代码编写



第一步先在根目录下运行'python mAP/get_ground_truth_txt.py'获得每张测试图片的GT数据

第二步在根目录下运行'python mAP/get_detection_result_txt.py'获取每张测试图片的推理数据

第三步在根目录下运行'python mAP/main.py'获得mAP相关的输出，所有的输出都在mAP/output目录下





