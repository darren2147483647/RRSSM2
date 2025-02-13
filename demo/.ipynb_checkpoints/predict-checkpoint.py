# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import sys
sys.path.insert(0, "./")
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot, my_show_result_pyplot
from mmseg.core.evaluation import get_palette
import glob
import os
import mmcv
from tqdm import tqdm
import time

def main():
    parser = ArgumentParser()
    parser.add_argument('--img_path', help='Image file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='drone',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=1,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    

    img_paths = sorted(glob.glob("./test_image_folder/*.jpg"))
    
    start_time = time.time()
    for img_path in tqdm(img_paths):
        # test a single image
        result = inference_segmentor(model, img_path)
        # show the results
        img  = my_show_result_pyplot(
            model,
            img_path,
            result,
            get_palette(args.palette),
            opacity=args.opacity)
        
        output_image = "./demo/output/" + img_path.split('/')[-1]
        mmcv.imwrite(img, output_image)
        
    print("prediction time:", time.time() - start_time)
    print("total frame:", len(img_paths))


if __name__ == '__main__':
    main()