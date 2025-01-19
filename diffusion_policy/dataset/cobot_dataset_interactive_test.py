# %%
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
root=zarr.open_group("/beegfs_hdd/data/nfs_share/share/share_datas/cobot_push_cube_dataset_collecting/training_data/push_cube_2025-01-16-jls")
replay_buffer = ReplayBuffer(root)
# %%
import sys
import numpy as np
sys.path.append('/beegfs_hdd/data/nfs_share/users/yinzi/nishome/cobot_diffusion_policy')
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

with hydra.initialize('../config'):
    cfg = hydra.compose('train_cobot_real_image_workspace')
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)

from matplotlib import pyplot as plt
normalizer = dataset.get_normalizer()
nactions = normalizer['action'].normalize(dataset.replay_buffer['action_2d'][:])
diff = np.diff(nactions, axis=0)
dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
_ = plt.hist(dists, bins=100);

# %%
i_data = iter(dataset).__next__()
