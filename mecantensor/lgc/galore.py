import numpy as np
class GaLoreTernaryProjector:
    def __init__(self, weight_shape, rank=128):
        self.shape = weight_shape
        self.rank = rank
        self.projection_left = None
    def project_gradient_down(self, raw_gradient):
        if self.projection_left is None:
            self.projection_left = np.random.randn(self.shape[0], self.rank).astype(np.float32)
            self.projection_left, _ = np.linalg.qr(self.projection_left)
        return self.projection_left.T @ raw_gradient
    def project_update_up(self, low_rank_update):
        return self.projection_left @ low_rank_update
def execute_galore_pretraining_step(weights, loss):
    grad = np.random.randn(*weights.shape).astype(np.float32)
    proj = GaLoreTernaryProjector(weights.shape, rank=min(128, min(weights.shape)))
    tiny_grad = proj.project_gradient_down(grad)
    scale = max(np.abs(tiny_grad).max(), 1e-8) / 127.0
    compressed = np.clip(np.round(tiny_grad / scale), -127, 127)
    update = compressed.astype(np.float32) * scale * 0.01
    full_update = proj.project_update_up(update)
    weights = weights - full_update
    gamma = max(np.abs(weights).mean(), 1e-5)
    return np.clip(np.round(weights / gamma), -1, 1).astype(np.float32)
