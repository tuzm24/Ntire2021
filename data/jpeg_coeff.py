import jpegio as jio
import torch
import torch_dct
import numpy as np

m_np = np.array([[1.0, 1.0, 1.0],
              [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
              [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

m_torch = torch.Tensor(([[1.0, 1.0, 1.0],
              [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
              [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]]))

def getJpegCoeff(path):
    jpeg = jio.read(path)
    return jpeg.coef_arrays, jpeg.quant_tables, jpeg.image_height, jpeg.image_width


def YUV2RGB_np(yuv):
    _rgb = np.dot(yuv, m_np)
    _rgb[:, :, 0] -= 179.45477266423404
    _rgb[:, :, 1] += 135.45870971679688
    _rgb[:, :, 2] -= 226.8183044444304
    _rgb = _rgb.clip(0, 255)
    return (_rgb + 0.5).astype('uint8')

# (B,3,H,W) for YUV420
def inverseDCT(coef_list, quant_tables):
    for ch in range(3):
        torch.stack([coef_list[ch][:,:,i::8, j::8] for i in range(8) for j in range(8)], dim=2)

