
from segmentation_models_pytorch import create_model
from lightning import LightningModule
import segmentation_models_pytorch as smp
import torch
from typing import Dict, Any



# add return type annotation as None
class unet_clothes_segment(LightningModule):
    def __init__(self,
                 arch : str ,
                 encoder_name : str,
                 encoder_weights : str,
                 in_channels : int,
                 classes : int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,

                 ) -> None:
        super().__init__()

        self.metrics = []

        self.save_hyperparameters()

        # ============================================================
        #                         setup the model
        # ============================================================
        self.model = create_model(arch=self.hparams.arch,
                                  encoder_name=self.hparams.encoder_name,
                                  encoder_weights=self.hparams.encoder_weights,
                                  in_channels=self.hparams.in_channels,
                                  classes=self.hparams.classes,

                                  )
        # ============================================================
        #                         setup the loss
        # ============================================================
        self.loss_function = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # ============================================================
        #                         preprocessing to apply
        # ============================================================
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(self.hparams.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # apply the standardization
        x = (x - self.mean) / self.std
        # pass via the model
        return self.model(x)

    def training_step(self, batch : torch.Tensor, batch_idx : int)-> dict:
        # get the images and the labels
        images, labels = batch['cloth'], batch['mask_cloth']

        # assert the the labels is holding only 0 and 1 values
        assert torch.unique(labels).tolist() == [0, 1]
        # get the predictions
        preds = self(images)
        # calculate the loss
        loss = self.loss_function(preds, labels)

        # apply thresholding
        prob_mask = preds.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), labels.long(), mode="binary")

        # will be used in the on_train_epoch_end hook
        self.metrics.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

        self.log_dict(
            {
                "train_loss": loss.item(),

            }, prog_bar=True, logger=True
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def on_train_epoch_end(self)-> None:

        outputs = self.metrics
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"train_per_image_iou": per_image_iou,
            f"train_dataset_iou": dataset_iou,
        }

        self.metrics.clear()
        self.log_dict(metrics, prog_bar=True, logger=True)

    def configure_optimizers(self)-> Dict[str, Any]:


        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}





