from pathlib import Path
import numpy as np
from PIL import Image
from src.data import is_empty_mask

def test_mask_ground_truth(tmp_path: Path):
    a=tmp_path/'a.png'; b=tmp_path/'b.png'; Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(a); x=np.zeros((8,8),dtype=np.uint8); x[3,3]=255; Image.fromarray(x).save(b); assert is_empty_mask(a); assert not is_empty_mask(b)
