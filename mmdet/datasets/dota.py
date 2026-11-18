from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DotaDataset(CocoDataset):

    CLASSES = ('plane', 'baseball-diamond',
               'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle',
               'ship', 'tennis-court',
               'basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool',
               'helicopter')
