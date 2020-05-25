# EfficientDet_PyTorch


## Reference
1. 论文(paper):   
[https://arxiv.org/pdf/1911.09070.pdf](https://arxiv.org/pdf/1911.09070.pdf)  

2. 代码参考(reference code):  
[https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

3. EfficientNet 主干网代码来源(Backbone code source):  
[https://github.com/Jintao-Huang/EfficientNet_PyTorch](https://github.com/Jintao-Huang/EfficientNet_PyTorch)  

4. 预训练模型来自(The pre-training model comes from):  
[https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)   
因为修改了模型，所以我把预训练模型的state_dict进行了重组，并进行发布  
(Because I changed the model, I reorganized the state_dict for the pretraining model and release it)  


权重见 release. 或在百度云中下载:  
链接：[https://pan.baidu.com/s/1VrO0eBmSHlB8_haEJ7WbuA](https://pan.baidu.com/s/1VrO0eBmSHlB8_haEJ7WbuA)   
提取码：2kq9  



## 使用方式(How to use)

#### 1. 预测图片(Predict images)
```
python3 pred_image.py
```

#### 2. 预测视频(Predict video)
```
python3 pred_video.py
```

#### 3. 简单的训练案例(Simple training cases)
```
python3 easy_examples.py
```

## 性能 
如果打不开可在`images/`与`docs/`文件夹中查看  

![性能](./docs/性能对比可视化.png)

#### d0效果

![原图片](./images/1.png)

![检测图片](./images/1_d0.png)
