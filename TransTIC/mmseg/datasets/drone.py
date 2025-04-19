from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DroneDataset(CustomDataset):

    ##3類別
    CLASSES = (
        'background', 'road','river')

    PALETTE = [[0, 0, 0], [255, 255, 255],[128, 128,128]] 
    ##3類別
    
    ##2類別
#     CLASSES = (
#         'background', 'target')

#     PALETTE = [[0, 0, 0], [255, 255, 255]] 
    ##2類別

    def __init__(self, **kwargs):
        super(DroneDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
