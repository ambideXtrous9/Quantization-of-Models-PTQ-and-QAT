# Quantization 

Link : [Quantization : PTQ and QAT on CNN using Keras](https://www.kaggle.com/code/sushovansaha9/quantization-ptq-and-qat-on-cnn-using-keras/notebook)

#### **Quantization is a model size reduction technique that converts model weights from high-precision floating-point representation to low-precision floating-point (FP) or integer (INT) representations, such as 16-bit or 8-bit.**

![image](https://github.com/ambideXtrous9/Quantization-PTQ-and-QAT/assets/31372586/9e190c2f-7f8e-4100-8c02-927c91a19364)

![image](https://github.com/ambideXtrous9/Quantization-PTQ-and-QAT/assets/31372586/8fe19492-4202-4b70-bc7a-d8e87ddfb910)

### **Post-Training Quantization (PTQ)**

Post-training quantization (PTQ) is a quantization technique where the model is quantized after it has been trained.

### **Quantization-Aware Training (QAT)**

Quantization-aware training (QAT) is a fine-tuning of the PTQ model, where the model is further trained with quantization in mind. The quantization process (scaling, clipping, and rounding) is incorporated into the training process, allowing the model to be trained to retain its accuracy even after quantization

![image](https://github.com/ambideXtrous9/Quantization-PTQ-and-QAT/assets/31372586/1abcf893-767d-4331-b5d9-1f83b1727bb3)


#### References :

1. [QAT PyTorch](https://github.com/fbsamples/pytorch-quantization-workshop)
2. [QAT Details](https://towardsdatascience.com/inside-quantization-aware-training-4f91c8837ead)
