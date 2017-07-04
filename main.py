import code_scaner_cnn
from PIL import Image

image = Image.open('code.jpg')
code = code_scaner_cnn.scaner(image)
print code
