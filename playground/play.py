from basicsr.utils import bgr2ycbcr
from PIL import Image
import numpy as np
import torch
butterfly = Image.open('datasets/Set5/LRbicx3/butterfly.png')
scale = 3
butterfly = butterfly.resize((butterfly.width * scale, butterfly.height * scale), Image.BICUBIC)
butterfly = np.array(butterfly)
butterfly = bgr2ycbcr(butterfly, y_only=True)
#save image
butterfly = Image.fromarray(butterfly)
butterfly.save('playground/butterfly.png')