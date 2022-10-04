from basicsr.utils import bgr2ycbcr
from PIL import Image
import numpy as np
import torch
butterfly = Image.open('butterfly.png')