from logging import root
from pathlib import Path
import numpy as np

from processing import build_data

import pandas as pd
from PIL import Image
import os
import yaml

#################################################################
def create_yolo_dataset_files(rebuild=False, generate_images=True, generate_labels=True, img_size=512, limit_records=None):
        df, ids = build_data(rebuild, img_size=img_size, limit_records=limit_records)
        test_fraction = 0.1
        test_split_index = round(test_fraction * len(ids))

        np.random.seed(42)
        np.random.shuffle(ids)
        dataset_name = f"dwg{img_size}"
        root_path = Path(f"data/{dataset_name}")
        root_path.mkdir(parents=True,exist_ok=True)

        train_images_path = root_path / "images/train"
        train_images_path.mkdir(parents=True, exist_ok=True)
        train_labels_path = root_path/ "labels/train"
        train_labels_path.mkdir(parents=True, exist_ok=True)
        train_desc_file_path = root_path / "train.txt"

        val_images_path =  root_path /"images/val"
        val_images_path.mkdir(parents=True, exist_ok=True)
        val_labels_path =  root_path / "labels/val"
        val_labels_path.mkdir(parents=True, exist_ok=True)
        val_desc_file_path = root_path / "val.txt"

        # folder for images with plotted annotations
        test_images_path = root_path / "images/test"
        test_images_path.mkdir(parents=True, exist_ok=True)

        max_labels = 0

        with open(train_desc_file_path, "w") as train_desc_file:
                with open(val_desc_file_path, "w") as val_desc_file:
                        for i, group_id in enumerate(ids):
                                desc_file = train_desc_file
                                image_folder = str(train_images_path)
                                label_folder = str(train_labels_path)
                                if i < test_split_index:
                                        desc_file = val_desc_file
                                        image_folder = str(val_images_path)
                                        label_folder = str(val_labels_path)

                                image_format = 'png'
                                image_file_name = "{}/{}.{}".format(image_folder, group_id, image_format)
                                label_file_name = "{}/{}.txt".format(label_folder, group_id)
                                image_with_annotations_file_name = "{}/{}.{}".format(str(test_images_path), group_id, image_format)
                                if generate_images:
                                        #source_img_stripped = df[(df['GroupId'] == group_id)]['StrippedFileName'].iloc[0]
                                        #source_image_annotated = df[(df['GroupId'] == group_id)]['AnnotatedFileName'].iloc[0]
                                        source_image_annotated = f'./data/images/annotated_{group_id}.png'
                                        def ResaveAsSquareSize(src_img_path, trgt_img_path):
                                                '''
                                                https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
                                                '''
                                                src = Image.open(src_img_path)
                                                max_size = max(src.size)
                                                trg = Image.new("RGBA", (max_size, max_size))

                                                # http://espressocode.top/python-pil-paste-and-rotate-method/
                                                # image is pasted using top left coords
                                                # while drawing coordinate system 0,0 is bottom left
                                                # so we have to shift by y size difference
                                                trg.paste(src, (0, max_size - src.size[1]))
                                                trg.save(trgt_img_path)

                                        # ResaveAsSquareSize(source_img_stripped, image_file_name)
                                        # ResaveAsSquareSize(source_image_annotated, image_with_annotations_file_name)
                                        ResaveAsSquareSize(source_image_annotated, image_file_name)

                                desc_file.write("{}\n".format(image_file_name))

                                if generate_labels:
                                        dims = df[(df['GroupId'] == group_id) & (df['ClassName'] == 'AlignedDimension')]

                                        labels = []
                                        for _, dim_row in dims.iterrows():
                                                dim_x_coords = [dim_row['XLine1Point.X'], dim_row['XLine2Point.X'], dim_row['DimLinePoint.X']] 
                                                dim_y_coords = [dim_row['XLine1Point.Y'], dim_row['XLine2Point.Y'], dim_row['DimLinePoint.Y']] 

                                                x = min(dim_x_coords)
                                                y = min(dim_y_coords)
                                                bb_width = max(dim_x_coords) - x
                                                bb_height = max(dim_y_coords) - y

                                                bb_center_x = x + (bb_width / 2)
                                                bb_center_y = y + (bb_height / 2)

                                                vec_dim_x = dim_row['XLine2Point.X'] - dim_row['XLine1Point.X']
                                                vec_dim_y = dim_row['XLine2Point.Y'] - dim_row['XLine1Point.Y']

                                                # class will be dimension baseline direction,
                                                if vec_dim_x >= 0 and vec_dim_y >= 0:
                                                        lbl = 0 # top
                                                elif vec_dim_x >= 0 and vec_dim_y <= 0:
                                                        lbl = 1 # left
                                                elif vec_dim_x <= 0 and vec_dim_y >= 0:
                                                        lbl = 2 # btm
                                                elif vec_dim_x <= 0 and vec_dim_y <= 0:
                                                        lbl = 3 # right

                                                labels.append([lbl, bb_center_x, bb_center_y, bb_width, bb_height])

                                        with open(label_file_name, 'w') as label_file:
                                                for cat, center_x, center_y, bb_width, bb_height in labels:
                                                        label_file.write("{} {} {} {} {} \n".format(
                                                                cat,
                                                                center_x / img_size,
                                                                1 - center_y / img_size,
                                                                bb_width / img_size,
                                                                bb_height / img_size
                                                        ))

                                        if max_labels < len(labels):
                                                max_labels = len(labels)

        # save yaml
        dwg_dataset_dict = {
                "path": ".",
                "train": str(train_desc_file_path),
                "val": str(val_desc_file_path),
                "test": "",
                "nc": 4,   # number of classes
                "names": [ '++', '+-', '--', '-+' ]  }

        with open( f'data/{dataset_name}/{dataset_name}.yaml', 'w') as f:
                yaml.safe_dump(dwg_dataset_dict, f, sort_keys=False)

        print("Max labels per image: ", max_labels)

if __name__ == "__main__":
    create_yolo_dataset_files(rebuild=True, img_size=512, limit_records=1200)