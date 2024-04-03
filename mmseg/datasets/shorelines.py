# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Shorelines(BaseSegDataset):
    """NIBRS Shorelines dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes = (
        'background',
        'Man-made structure',   
        'Sandy beach',       
        'Rocky outcrop',      
        'Gravel beach',       
        'Block beach',       
        'Ice and snow',     
        'Clay beach',  
        'Wetland area'),
        palette = [
        [0,0,0],                # Black color for background
        [255, 0, 0],            # Red color for man-made structures
        [255, 255, 102],        # Light yellow color for sandy beaches
        [153, 102, 51],         # Brown color for rocky outcrops
        [204, 204, 204],        # Light gray color for gravel beaches
        [102, 102, 102],        # Dark gray color for block beaches
        [255, 255, 255],        # White color for ice and snow 
        [204, 153, 102],        # Light brown color for clay beaches
        [51, 153, 102]])        # Green color for wetland areas

    

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


