# 資料收集(2025/02/05之前)
## TensorFlow 物件偵測相關資源
- **即時物件偵測（GeeksforGeeks 教學）**  
  [Real-time Object Detection using TensorFlow](https://www.geeksforgeeks.org/real-time-object-detection-using-tensorflow/)
- **TensorFlow 物件偵測 GitHub**  
  [GitHub Repository - TensorFlow Models](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md)
- **物件偵測調查論文（2019）**  
  [A Survey on Object Detection](https://arxiv.org/pdf/1905.05055)
- **TensorFlow 官方網站**  
  [TensorFlow Tutorials](https://www.tensorflow.org/tutorials?hl=zh-tw)
- **使用預訓練模型來訓練新模型（GT Wang）**  
  [TensorFlow Object Detection API Custom Object Model Training Tutorial](https://blog.gtwang.org/programming/tensorflow-object-detection-api-custom-object-model-training-tutorial/)

## 深度學習與遷移學習相關資源
- **神經網路與深度學習（電子書網站）**  
  [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
- **遷移學習介紹（iT 邦幫忙）**  
  [遷移學習文章](https://ithelp.ithome.com.tw/articles/10279642)
- **遷移學習與預訓練模型（iT 邦幫忙）**  
  [遷移學習與預訓練模型](https://ithelp.ithome.com.tw/articles/10364503)

# 資料收集（2025/02/05）
## EfficientNet 相關資源
- **EfficientNet 說明（iT 邦幫忙）**  
  [EfficientNet 簡介](https://ithelp.ithome.com.tw/articles/10303001)
- **EfficientNet 詳細解讀（Medium）**  
  [EfficientNet 論文閱讀](https://medium.com/ching-i/efficientnet-%E8%AB%96%E6%96%87%E9%96%B1%E8%AE%80-e828ac005ce8)

## TensorFlow 安裝指南
- **TensorFlow 安裝與環境設置（Medium）**  
  [TensorFlow 安裝教學](https://medium.com/@zera.tseng888/tensorflow%E5%AE%89%E8%A3%9D%E8%88%87%E7%92%B0%E5%A2%83%E8%A8%AD%E5%AE%9A-e067db784e04)
- **TensorFlow 影片教學（YouTube）**  
  [TensorFlow 安裝與設定（Krish Naik）](https://www.youtube.com/watch?v=q5YCba5cVxQ&list=PLZoTAELRMXVNvTfHyJxPRcQkpV8ubBwHo&ab_channel=KrishNaik)

## DETR 相關資源（基於 PyTorch）
- **DETR 模型介紹（iT 邦幫忙）**  
  [DETR 介紹](https://ithelp.ithome.com.tw/articles/10327551)
- **DETR 預訓練模型（Hugging Face）**  
  [DETR ResNet-50（COCO 2017 Dataset）](https://huggingface.co/facebook/detr-resnet-50)
  > *注意：DETR 使用 PyTorch，可能與 TensorFlow 不太兼容，需進一步評估，看完的人可以再提出意見。*

## TensorFlow 自行訓練模型
- **Faster R-CNN Inception ResNet V2（Kaggle）**  
  [TensorFlow Faster R-CNN Inception ResNet V2](https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/tensorFlow2/640x640/1)

# TensorFlow安裝流程(2025/2/12)
**你可以依照2/5號的tensorflow安裝指南來做但請注意下載的版本可以參考後面的連結，[TensorFlow 2 Object Detection API tutorial documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)，因為目前他們已停止windows使用gpu的版本，只支援到tensorflow2.10.0的版本，想知道詳細的軟件配置可以參考這邊[Build from source on Windows](https://tensorflow.google.cn/install/source_windows)。**

**如果安裝cuda11.2有出現問題可以參考這幾篇文章：**
- [win11在已有cuda12.6情形下如何配置tensorflow-gpu_cuda12.6对应的tensorflow-CSDN博客](https://blog.csdn.net/2201_75372819/article/details/142532068?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-142532068-blog-119043543.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.1&utm_relevant_index=2)
- [配置Tensorflow使用CUDA进行GPU加速(超详细教程)_tensorflow cuda-CSDN博客](https://blog.csdn.net/m0_51302496/article/details/137185657)

**CUDA與Cudnn相關安裝連結**
- [CUDA Toolkit 11.2 Downloads](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
- [Cudnnv8.1.0(cuDNN Archive | NVIDIA Developer)](https://developer.nvidia.com/rdp/cudnn-archive)