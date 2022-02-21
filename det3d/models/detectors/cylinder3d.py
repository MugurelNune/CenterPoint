from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class Cylinder3D(SingleStageDetector):
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
    ):
        super(Cylinder3D, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        features = data['return_fea']
        grid_ind = data['grid_ind']
        batch_size = len(data['points'])

        coords, features_3d = self.reader(features, grid_ind)

        x = self.backbone(features_3d, coords, batch_size)

        if self.with_neck:
            x = self.neck(x)

        return x

    def forward(self, example, return_loss=True, **kwargs):
        x = self.extract_feat(example)
        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)
