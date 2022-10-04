from basicsr.utils import bgr2ycbcr
from PIL import Image
import numpy as np
import torch
butterfly = Image.open('datasets/Set5/LRbicx3/butterfly.png')