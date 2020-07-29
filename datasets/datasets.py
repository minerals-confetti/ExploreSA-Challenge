import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import transform, Proj
import random

# helper func to extract a window
from rasterio.windows import Window

def convert_from_EPSG4326(xy, dataset):
    """given a list of (lat, lon) encoded in EPSG:4326, returns a list of (x, y) encoded in dataset.crs"""
    inproj, outproj = Proj('EPSG:4326'), dataset.crs

    return list(zip(*transform(inproj, outproj, *zip(*xy))))

def extract_cubic(dataset, coords, size=(9, 9)):
    '''takes in a opened rasterio dataset, a list of tuples of coords [(lat, long),], size of the extracted cubics, 
    and returns coords_length * size1 * size2 * n_bands'''
    bands = dataset.count
    output = np.zeros((len(coords), *size, bands))

    trans_coords = convert_from_EPSG4326(coords, dataset)
    for i, (lon, lat) in enumerate(trans_coords):
        py, px = dataset.index(lon, lat)
        window = Window(px - size[0] // 2, py - size[1] // 2, size[0], size[1])
        clip = dataset.read(window=window)
        try:
            output[i] = np.transpose(clip, (1, 2, 0))
        except ValueError:
            output[i] = np.full((*size, bands), np.nan)

    return output

#transforms
class RandomRotation():
    def __init__(self):
        ''' adds random rotations, (quadruples data size)'''
        self.angles = [0, 90, 180, 270]

    def __call__(self, cubic):
        
        angle = random.choice(self.angles)
        if angle == 0:
            return cubic
        elif angle == 90:
            return cubic.transpose(1, 2).flip(2)
        elif angle == 180:
            return cubic.flip(2).flip(1)
        elif angle == 270:
            return cubic.transpose(1, 2).flip(1)

class LocationChecker():
    '''creates a LocationChecker from a directory of datasets, 
    and checks if a location (EPSG4326) {"LATITUDE": num, "LONGITUDE": num} is valid
    if valid, returns the dataset location, and if from multiple datasets, returns a random one
    if invaid, returns None
    '''

    def __init__(self, datadir, rigorous=True, size=(9, 9)):
        self.bounds_dict = {}
        for dataset in self.walkdir(datadir):
            with rasterio.open(dataset) as ds:
                llx, lly, urx, ury = ds.bounds
                self.bounds_dict[dataset] = self.convert_to_EPSG4326([(llx, lly), (urx, ury)], ds)
        self.rigorous = rigorous
        self.size = size

    def check(self, coords):
        '''coords: (EPSG4326) {"LATITUDE": num, "LONGITUDE": num}'''
        valid_list = []
        for dataset, bounds in self.bounds_dict.items():
            if bounds[0][0] < coords["LATITUDE"] < bounds[1][0]:
                if bounds[0][1] < coords["LONGITUDE"] < bounds[1][1]:
                    valid_list.append(dataset)

        if self.rigorous and valid_list:
            valider_list = []
            for dataset in valid_list:
                with rasterio.open(dataset) as ds:
                    sample = extract_cubic(ds, [(coords["LATITUDE"], coords["LONGITUDE"])], size=self.size)
                    if not np.isnan(np.sum(sample)) and sample.shape[1:3] == self.size:
                        valider_list.append(dataset)
            valid_list = valider_list

        if valid_list:
            return random.sample(valid_list, 1)[0]
        
        return None

    def walkdir(self, rootdir):
        outlist = []
        for root, _, files in os.walk(rootdir):
            for name in files:
                outlist.append(root + "/" + name)
        return outlist
    
    def convert_to_EPSG4326(self, xy, dataset):
        """given a list of (x, y) encoded in dataset.crs, returns a list of (x, y) encoded in EPSG4326"""

        inproj, outproj = dataset.crs, Proj('EPSG:4326')

        return list(zip(*transform(inproj, outproj, *zip(*xy))))

    def checkdf(self, csv, outputfile):
        '''checks a csv with a "LATITUDE", and a "LONGITUDE" column, removes all rows that aren't in the dataset, adds path to given image'''
        
        df = pd.read_csv(csv, index_col=False)
        columns = df.columns
        if "LATITUDE" not in columns or "LONGITUDE" not in columns:
            raise Exception("LATITUDE, and/or LONGITUDE not in csv")
        df_dict = df.to_dict(orient="list")

        keepindicies = []
        paths = []
        for i, (lat, lon) in enumerate(zip(df_dict["LATITUDE"], df_dict["LONGITUDE"])):
            file_path = self.check(self.conv_coords((lat, lon)))
            if file_path:
                keepindicies.append(i)
                paths.append(file_path)
            else:
                print("not keeping {}".format(i))
            
        outputdf = df.iloc[keepindicies]
        outputdf["paths"] = paths
        outputdf.to_csv(outputfile, index=False)

    def conv_coords(self, coords, reverse=False):
        '''converts (lat, lon) to coords used by this class'''
        if reverse:
            return coords["LATITUDE"], coords["LONGITUDE"]
        return {"LATITUDE": coords[0], "LONGITUDE": coords[1]}

class FDSSCDataset(Dataset):
    '''Creates a dataset from a directory of datasets, and a csv of VALID locations, and crops out a location to feed into FDSSC

    The csv should have a "label" column, "LATITUDE", and "LONGITUDE"
    '''

    def __init__(self, datadir, cachedir, csv, size=(9, 9), transform=[], lochecker=None):
        self.locations = pd.read_csv(csv)
        #checking if csv is valid
        self.lbound = None
        columns = self.locations.columns
        if "LATITUDE" not in columns or "LONGITUDE" not in columns or "label" not in columns or "paths" not in columns:
            raise Exception("csv invalid, LATITUDE, LONGITUDE, label, paths must be in a valid csv")

        if lochecker:
            self.locationChecker = lochecker
        else:
            self.locationChecker = LocationChecker(datadir, rigorous=False)
        
        self.cachedir = cachedir
        self.size = size
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = int(self.locations.loc[idx, "label"])
        coords = {"LATITUDE": float(self.locations.loc[idx, "LATITUDE"]), "LONGITUDE": float(self.locations.loc[idx, "LONGITUDE"])}

        dataset_dir = self.locations.loc[idx, "paths"]
        try:
            cubic = np.load("{}/{}_{}_{}.npy".format(self.cachedir, coords["LATITUDE"], coords["LONGITUDE"], dataset_dir.split("/")[-1].split(".")[0]), allow_pickle=True)
        except FileNotFoundError:
            cubic = self.extract_cubic(dataset_dir, self.locationChecker.conv_coords(coords, reverse=True), size=self.size)
            np.save("{}/{}_{}_{}.npy".format(self.cachedir, coords["LATITUDE"], coords["LONGITUDE"], dataset_dir.split("/")[-1].split(".")[0]), cubic)
        
        if self.lbound != None:
            cubic = self.normOP(cubic)

        cubic = torch.tensor(cubic)

        for transform in self.transform:
            cubic = transform(cubic)
        
        sample = {"label": label, "image": cubic}

        return sample
    
    def __len__(self):
        return len(self.locations)

    def extract_cubic(self, dataset_dir, coords, size=(9, 9)):
        '''takes in a dataset dir, a tuple of coord (lat, long), size of the extracted cubics, 
        and returns coords_length * size1 * size2 * n_bands'''

        with rasterio.open(dataset_dir) as dataset:
            trans_coords = convert_from_EPSG4326([coords], dataset)
            lon, lat = trans_coords[0]
            py, px = dataset.index(lon, lat)
            window = Window(px - size[0] // 2, py - size[1] // 2, size[0], size[1])
            clip = dataset.read(window=window)
            output = np.transpose(clip, (1, 2, 0))

        return np.expand_dims(output, axis=0)

    def getMinMax(self):
        max_value = self[0]["image"].numpy().max(axis=(0, 1, 2), keepdims=False)
        min_value = self[0]["image"].numpy().min(axis=(0, 1, 2), keepdims=False)

        for i in range(1, len(self)):
            temp_max = self[i]["image"].numpy().max(axis=(0, 1, 2), keepdims=False)
            temp_min = self[i]["image"].numpy().min(axis=(0, 1, 2), keepdims=False)

            max_value = np.where(temp_max > max_value, temp_max, max_value)
            min_value = np.where(temp_max < min_value, temp_min, min_value)

        self.min_value = min_value
        self.max_value = max_value

        return min_value, max_value

    def normalize(self, lbound, ubound, min_value=None, max_value=None):
        if min_value is not None:
            self.min_value = min_value
        if max_value is not None:
            self.max_value = max_value

        self.lbound = lbound
        self.ubound = ubound

    def normOP(self, cubic):
        
        return (self.ubound - self.lbound) * (cubic - np.expand_dims(self.min_value, axis=(0, 1, 2))) / np.expand_dims((self.max_value - self.min_value), axis=(0, 1, 2)) + self.lbound
