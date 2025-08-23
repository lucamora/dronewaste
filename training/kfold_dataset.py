import os
import json
import yaml
import random
import shutil
from datetime import datetime
from skmultilearn.model_selection import iterative_train_test_split
from scipy.sparse import csr_matrix
import numpy as np
from abc import ABC, abstractmethod


class SitesGroundTruth:
    def __init__(self, gt_path):
        with open(gt_path, 'r') as f:
            self.gt = json.load(f)

    @property
    def images(self):
        return self.gt['images']

    @property
    def categories(self):
        return self.gt['categories']

    def get_sites(self):
        sites = set([i['site'] for i in self.gt['images']])
        return list(sites)

    def get_images_by_sites(self, sites):
        """
        Filter images from the specified sites
        """
        # if sites is a string -> convert to list
        if isinstance(sites, str):
            sites = [sites]

        # return images for the specified sites
        return [img['file_name'] for img in self.gt['images'] if img['site'] in sites]

    def collect_sites_data(self, sites):
        """
        Returns images and annotations for the specified sites
        """
        # if sites is a string -> convert to list
        if isinstance(sites, str):
            sites = [sites]

        # get images from the specified sites
        images = [i for i in self.gt['images'] if i['site'] in sites]

        # get annotations of the selected images
        img_ids = [img['id'] for img in images]
        annotations = [a for a in self.gt['annotations'] if a['image_id'] in img_ids]

        return images, annotations


class SitesDatasetSplit:
    def __init__(
        self,
        dataset_path,
        output_path,
        format,
        timestamp=None,
        copy_images=False,
        no_empty=False,
    ):
        path_parts = dataset_path.split('/')
        self.in_base_path = '/'.join(path_parts[:-2])
        self.version = path_parts[-2]
        self.in_dataset_path = dataset_path
        self.out_base_path = output_path

        # save current timestamp
        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        else:
            self.timestamp = timestamp

        # load ground truth
        self.gt = SitesGroundTruth(self.in_dataset_path)

        # get sites
        self.sites = self.gt.get_sites()

        # list of images of the splits
        self.images = {'train': [], 'val': [], 'test': []}

        # create exporter
        imgs_path = os.path.join(self.in_base_path, 'images')
        if format == 'coco':
            self.exporter = CocoExporter()
        elif format == 'yolo':
            self.exporter = YoloExporter()
        else:
            raise ValueError(f'Invalid export format: {format}')

        # initialize exporter
        self.exporter.init(
            self.gt.categories,
            imgs_path,
            self.export_path,
            copy_images,
            no_empty,
        )

    @property
    def export_path(self):
        # otherwise create a dedicated export folder
        dataset_id = f'{self.version}_{self.timestamp}'
        return os.path.join(self.out_base_path, dataset_id)

    @property
    def dataset_file(self):
        # valid only for yolo format
        return os.path.join(self.export_path, f'dataset.yaml')

    @property
    def images_path(self):
        # valid only a coco format
        return os.path.join(self.export_path, 'images')

    @property
    def train_file(self):
        # valid only a coco format
        return self.__split_file('train')

    @property
    def val_file(self):
        # valid only a coco format
        return self.__split_file('val')

    @property
    def test_file(self):
        # valid only a coco format
        return self.__split_file('test')

    def __split_file(self, split):
        # valid only for coco format
        return os.path.join(self.export_path, f'{split}.json')

    def __get_split_sites(self, include, exclude):
        """
        Returns all the sites for a dataset split.
        By default filters the sites in the "include" list.
        If "include" is 'all', returns all the sites except the ones in the "exclude" list.
        """

        # collect the list of included sites
        # if include is 'all' -> get all sites except the ones in the exclude list
        if include == 'all':
            exclude = exclude[0] + exclude[1]  # exclude has exactly two elements
            include = [s for s in self.sites if s not in exclude]

        return include

    def __export_split(self, split, images, annotations):
        # create split
        self.exporter.create_split(split)

        # add site to split
        self.images[split].extend([image['file_name'] for image in images])
        self.exporter.add_site(images, annotations)

        # export split
        self.exporter.export_split(self.__split_file(split))

        # update stats
        images_count = len(images)
        annots_count = len(annotations)

        return images_count, annots_count

    def __create_info_file(self):
        info = {
            'version': self.version,
            'num_classes': len(self.gt.categories),
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_dataset': self.in_dataset_path,
        }
        with open(os.path.join(self.export_path, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, sort_keys=False)

    def __stratified_split(self, images, annotations, ratio):
        image_ids = [i['id'] for i in images]
        category_ids = [c['id'] for c in self.gt.categories]

        n_images = len(image_ids)
        n_categories = len(category_ids)

        # create mapping for quick lookups
        image_id_to_idx = {id: idx for idx, id in enumerate(image_ids)}
        category_id_to_idx = {id: idx for idx, id in enumerate(category_ids)}

        # initialize sparse matrix for labels
        rows = []
        cols = []
        data = []

        # fill the label matrix
        for ann in annotations:
            img_idx = image_id_to_idx[ann['image_id']]
            cat_idx = category_id_to_idx[ann['category_id']]
            rows.append(img_idx)
            cols.append(cat_idx)
            data.append(1)

        X = csr_matrix((data, (rows, cols)), shape=(n_images, n_categories))
        y = X  # in multi-label case, X and y are the same

        # create sample index matrix
        sample_indices = np.array([[i] for i in range(n_images)])

        # set seed for random splitting
        random.seed(1337)
        np.random.seed(1337)

        # perform stratified split
        train_indexes, _, test_indexes, _ = iterative_train_test_split(
            sample_indices, y, test_size=ratio
        )

        # convert back to image IDs
        train_image_ids = [image_ids[idx[0]] for idx in train_indexes]
        val_image_ids = [image_ids[idx[0]] for idx in test_indexes]

        return train_image_ids, val_image_ids

    def split(self, train=None, val=None, test=None):
        """
        Splits sites into train, val, test
        - kfold: split sites in (train_val, test) -> train and val are later randomly split
        - master: split sites in (train, val) -> val has fixed sites
        - inference: define test sites -> train and val are not used
        """

        train = train or []
        val = val or []
        test = test or []
        train_set = self.__get_split_sites(include=train, exclude=[val, test])
        val_set = self.__get_split_sites(include=val, exclude=[train, test])
        test_set = self.__get_split_sites(include=test, exclude=[train, val])

        return train_set, val_set, test_set

    def create(self, train=None, val=None, test=None, debug=False):
        """
        Generates a dataset based on the defined splits.
        """

        if not debug:
            # export dataset
            self.exporter.export_dataset(self.dataset_file)
            # create info file
            self.__create_info_file()

        train_stats, val_stats, test_stats = None, None, None

        # if train and val are defined
        if train and val:
            if isinstance(val, float):
                # val is a number -> split trainval into train and val by the specified ratio
                trainval_imgs, trainval_annots = self.gt.collect_sites_data(train)

                # perform stratified split
                train_img_ids, val_img_ids = self.__stratified_split(
                    trainval_imgs,
                    trainval_annots,
                    val,  # val is the split ratio
                )

                # collect training data
                train_imgs = [i for i in trainval_imgs if i['id'] in train_img_ids]
                train_annots = [
                    a for a in trainval_annots if a['image_id'] in train_img_ids
                ]

                # collect validation data
                val_imgs = [i for i in trainval_imgs if i['id'] in val_img_ids]
                val_annots = [
                    a for a in trainval_annots if a['image_id'] in val_img_ids
                ]
            else:
                # val is a list of sites -> get train and val data by site
                train_imgs, train_annots = self.gt.collect_sites_data(train)
                val_imgs, val_annots = self.gt.collect_sites_data(val)

            # create train and val folders
            if not debug:
                train_stats = self.__export_split('train', train_imgs, train_annots)
                val_stats = self.__export_split('val', val_imgs, val_annots)

        # if test is defined
        if test:
            # get test data
            test_imgs, test_annots = self.gt.collect_sites_data(test)

            # create test folder
            if not debug:
                test_stats = self.__export_split('test', test_imgs, test_annots)

        if debug:
            return train_imgs, val_imgs, test_imgs
        return train_stats, val_stats, test_stats

    def delete(self):
        shutil.rmtree(self.export_path, ignore_errors=True)

    def get_images_by_split(self, split):
        return self.images[split]

    def get_images_by_sites(self, sites=None):
        # return all images
        if sites is None:
            return self.gt.images

        # filter images by sites
        return self.gt.get_images_by_sites(sites)


class DatasetExporter(ABC):
    def init(self, categories, imgs_path, export_path, copy_images, no_empty):
        self.imgs_path = imgs_path
        self.export_path = export_path
        self.copy_images = copy_images
        self.categories = categories
        self.no_empty = no_empty

        self.category_ids = [c['id'] for c in categories]
        self.category_names = [c['name'] for c in categories]

    def _export_image(self, src, dst):
        if self.copy_images:
            # copy image to export folder
            shutil.copy(src, dst)
        else:
            # create symbolic link to image in export folder
            os.symlink(src, dst)

    def export_dataset(self, dataset_file):
        os.makedirs(self.export_path, exist_ok=True)
        os.makedirs(os.path.join(self.export_path, 'images'), exist_ok=True)

    def create_split(self, split):
        self.split = split

    @abstractmethod
    def add_site(self, images, annotations):
        pass

    def export_split(self, split_file):
        pass


class CocoExporter(DatasetExporter):
    def create_split(self, split):
        super().create_split(split)

        # generate a json file for the set
        self.content = {
            'images': [],
            'annotations': [],
            'categories': self.categories,
        }

    def add_site(self, images, annotations):
        self.content['images'].extend(images)
        self.content['annotations'].extend(annotations)

        for image in images:
            img_filename = image['file_name']
            # export image
            self._export_image(
                src=os.path.join(self.imgs_path, img_filename),
                dst=os.path.join(self.export_path, 'images', img_filename),
            )

    def export_split(self, split_file):
        # save json file for the set
        with open(split_file, 'w') as f:
            json.dump(self.content, f, indent=2)


class YoloExporter(DatasetExporter):
    def export_dataset(self, dataset_file):
        super().export_dataset(dataset_file)

        content = {
            'path': self.export_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.category_names),  # number of classes
            'names': self.category_names,  # class names
        }
        # save dataset file
        with open(dataset_file, 'w') as f:
            yaml.safe_dump(content, f, sort_keys=False)

    def create_split(self, split):
        super().create_split(split)

        # create folders for the set
        os.makedirs(os.path.join(self.export_path, 'images', self.split), exist_ok=True)
        os.makedirs(os.path.join(self.export_path, 'labels', self.split), exist_ok=True)

    def add_site(self, images, annotations):
        for image in images:
            img_filename = image['file_name']

            # convert image annotations to yolo format
            yolo_annots = [
                self.__convert_bbox_to_yolo(annot, image['width'], image['height'])
                for annot in annotations
                if annot['image_id'] == image['id']
            ]

            if not self.no_empty or self.split != 'train' or len(yolo_annots) > 0:
                # export image
                self._export_image(
                    src=os.path.join(self.imgs_path, img_filename),
                    dst=os.path.join(self.export_path, 'images', self.split, img_filename),
                )

                # export annotations in yolo format
                annots_file = img_filename.rsplit('.', 1)[0] + '.txt'
                annots_path = os.path.join(
                    self.export_path, 'labels', self.split, annots_file
                )
                with open(annots_path, 'w') as f:
                    f.writelines(ann + '\n' for ann in yolo_annots)

    def __convert_bbox_to_yolo(self, annot, image_width, image_height):
        # convert bbox to yolo format
        x, y, w, h = annot['bbox']
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        w = w / image_width
        h = h / image_height
        # get index of the bbox category
        class_id = self.category_ids.index(annot['category_id'])
        return f'{class_id} {x_center} {y_center} {w} {h}'
