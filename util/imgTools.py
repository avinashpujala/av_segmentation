import cv2
import numpy as np

# flow_params = dict(flow=None, pyr_scale=0.5, levels=3, winsize=5,
#                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def estimate_optical_flow(imgs, flow_params=[None, 0.5, 3, 5, 3, 5, 1.2, 0]):
    """
    Estimates optic flow using cv2.calcOpticalFlowFarneback.
    Parameters
    ----------
    imgs: array, (nImgs, *imgDims)
        Images
    flow_params: list-like
        Flow parameters. Values based on an example presented online
    Returns
    -------
    mag: array, imgs.shape
        Magnitude of flow
    ang: array, imgs.shape
        Angle of flow
    """
    mag, ang = np.zeros_like(imgs), np.zeros_like(imgs)
    for i in range(1, len(imgs)):
        flow = cv2.calcOpticalFlowFarneback(imgs[i-1], imgs[i], *flow_params)
        mag[i-1], ang[i-1] = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag[i] = mag[i-1].copy()
    ang[i] = ang[i-1].copy()
    return mag, ang


