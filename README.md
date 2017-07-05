# Code_Scaner
使用 Tensorflow 识别教务管理系统验证码
Python、CNN

使用方法：

```
  import code_scaner_cnn
  from PIL import Image

  image = Image.open('code.jpg')
  code = code_scaner_cnn.scaner(image)
  print code
```
