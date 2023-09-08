
from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
import cv2
import numpy as np
from src import utils
from pathlib import Path
from rich.console import Console
from torchvision import transforms
from tqdm import tqdm
log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    with Console().status(f"loading model ckpt from {cfg.ckpt_path}", spinner='material') as status:

        log.info(f"Instantiating model <{cfg.model._target_}>")
        # load from ckpt
        model: LightningModule = hydra.utils.instantiate(cfg.model).load_from_checkpoint(cfg.ckpt_path)





        status.update(f"loading images to apply transform", spinner='aesthetic')
        # ============================================================
        #                  image transform to apply
        # ============================================================
        image2predict = list(Path(cfg.images_folder).iterdir())
        Console().log(f"found {len(image2predict)} images to predict", style='green')

        if len(image2predict) > 0:
            for k in tqdm(image2predict, colour='green'):
                mask_cloth = cv2.imread(k.as_posix(), cv2.IMREAD_GRAYSCALE)
                mask_cloth = np.array(mask_cloth)  # dtype = uint8
                image_name_save = k.name
                ret, thresh = cv2.threshold(mask_cloth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask_cloth = transforms.ToTensor()(thresh)  # dtype = float32, unique values = 0, 1, shape = (1, H, W)
                # include the batch dim.
                mask_cloth = mask_cloth.unsqueeze(0).to(model.device)
                outs = model(mask_cloth).sigmoid_()
                out_mask = (outs > 0.5).float()
    #             save as pil image
                transforms.functional.to_pil_image(out_mask.squeeze(0).squeeze(0)).save(Path(cfg.save_dir) / image_name_save)


    object_dict = {
        "cfg": cfg,
        "save_dir": cfg.save_dir,
        "model": model,

    }


    return object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    predict(cfg)


if __name__ == "__main__":
    main()
