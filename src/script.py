import subprocess
import os
import pathlib
import argparse
import shutil
import random
import uuid

from glob import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

SORT_DIR = '/home/r1_sorted_3Cla'
SRC_PATH = '/home/DeepPATH/DeepPATH_code'


def create_tiles(input_path):
    files = glob(f'{input_path}/**/*.svs', recursive=True)

    for f in files:
        print(f'File: {f}')

        proc = subprocess.run(f'python3 {SRC_PATH}/00_preprocessing/0b_tileLoop_deepzoom4.py -s 512 -e 0 -j 16 -B 50 '
                              f'-M 20 -o 512px_Tiled {f}', shell=True, stdout=subprocess.DEVNULL)

        if proc.returncode != 0:
            raise Exception(f'Tile creation for file: {f} failed with return code: {proc.returncode}')


def sort_jpegs(file_path, label_type, input_path, output_path):
    with open(file_path) as f:
        filenames = f.readlines()

    # Get names of unique cases
    patients = list(set([x[0:12] for x in filenames]))

    # Train-val-test split
    random.shuffle(patients)
    splits = [0.7, 0.9]

    patient_split = {'train': patients[0:int(splits[0] * len(patients))],
                     'valid': patients[int(splits[0] * len(patients)):int(splits[1] * len(patients))],
                     'test': patients[int(splits[1] * len(patients)):]}

    slides = []
    for split in ['train', 'valid', 'test']:
        pathlib.Path(f'{output_path}/{split}/{label_type}').mkdir(parents=True, exist_ok=True)
        for patient in patient_split[split]:
            for image in glob(f'{input_path}/{patient}*_files'):
                image_str = os.path.basename(image).split('_')[0]
                slides.append(image_str)

                for tile in glob(f'{image}/*/*.jpeg'):
                    tile_str = os.path.basename(tile)
                    new_str = '_'.join([split, image_str, tile_str])
                    shutil.copy(tile, os.path.join(output_path, split, label_type, new_str))

    return slides


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label, slide_string):
    if slide_string:
        feature = {
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
            'slide_string': _bytes_feature(slide_string)
        }
    else:
        feature = {
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_image_dict(data_path, slide=None):
    tcga_image_labels = {}
    label = 0
    folders = os.listdir(data_path)
    print(folders)

    for fol in folders:
        files = glob(os.path.join(data_path, fol, f'*{slide}*')) if slide else os.listdir(os.path.join(data_path, fol))
        for file in files:
            path = os.path.join(fol, file)
            tcga_image_labels[path] = label
        label += 1

    return tcga_image_labels


def generate_tf_records(base_folder, input_files, output_file, n_image, slide=None):
    record_file = output_file

    count = n_image
    with tf.io.TFRecordWriter(record_file) as writer:
        while count:
            filename, label = random.choice(input_files)
            temp_img = plt.imread(os.path.join(base_folder, filename))
            if temp_img.shape != (512, 512, 3):
                continue
            count -= 1

            image_string = np.float32(temp_img).tobytes()
            slide_string = slide.encode('utf-8') if slide else None
            tf_example = image_example(image_string, label, slide_string)
            writer.write(tf_example.SerializeToString())


def convert_to_tf_records(channel_jpegs_path, channel_name, op_dir, n_image, slides):
    if channel_name in ['train', 'valid']:
        tcga_image_labels = generate_image_dict(channel_jpegs_path)

        tot_files = len(tcga_image_labels)
        print(f'Channel: {channel_name}. Total number of images: {tot_files}')
        if tot_files == 0:
            return

        pathlib.Path(f'{op_dir}/{channel_name}').mkdir(parents=True, exist_ok=True)
        output_file = f'{op_dir}/{channel_name}/image-{str(uuid.uuid4())}.tfrecords'
        n_image = n_image if channel_name == 'train' else tot_files
        print(f'Count of input images: {n_image}')

        generate_tf_records(channel_jpegs_path, list(tcga_image_labels.items()), output_file, n_image)
    else:
        for slide in slides:
            tcga_image_labels = generate_image_dict(channel_jpegs_path, slide)

            tot_files = len(tcga_image_labels)
            print(f'Channel: {channel_name}. Total number of images for slide {slide}: {tot_files}')
            if tot_files == 0:
                continue

            pathlib.Path(f'{op_dir}/{channel_name}').mkdir(parents=True, exist_ok=True)
            output_file = f'{op_dir}/{channel_name}/{slide}.tfrecords'
            n_image = tot_files
            print(f'Count of input images: {n_image}')

            generate_tf_records(channel_jpegs_path, list(tcga_image_labels.items()), output_file, n_image, slide)


def main(n_images):
    input_path = '/opt/ml/processing/input'
    jpeg_path = '/home/512px_Tiled'
    sorted_jpeg_path = "/home/tcga-svs-tiled-sorted"
    output_path = f'/opt/ml/processing/output'

    tcga_svs_labels_path = '/home/tcga-svs-labels'
    tcga_svs_luad_file_path = f'{tcga_svs_labels_path}/tcga-lung-luad.txt'
    tcga_svs_lusc_file_path = f'{tcga_svs_labels_path}/tcga-lung-lusc.txt'
    tcga_svs_norm_file_path = f'{tcga_svs_labels_path}/tcga-lung-normal.txt'

    print('Creating tiles .....')
    create_tiles(input_path)
    print('Tiles created.')

    luad_slides = sort_jpegs(tcga_svs_luad_file_path, 'luad', jpeg_path, sorted_jpeg_path)
    lusc_slides = sort_jpegs(tcga_svs_lusc_file_path, 'lusc', jpeg_path, sorted_jpeg_path)
    norm_slides = sort_jpegs(tcga_svs_norm_file_path, 'normal', jpeg_path, sorted_jpeg_path)
    print('Images sorted.')

    slides = luad_slides + lusc_slides + norm_slides

    for channel in ['train', 'test', 'valid']:
        convert_to_tf_records(f'{sorted_jpeg_path}/{channel}', channel, output_path, n_images, slides)

    print('TF records conversion complete.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tiling images.')
    parser.add_argument('n_images', type=int, help='Number of images in the final TF record.')
    args = parser.parse_args()

    main(args.n_images)
