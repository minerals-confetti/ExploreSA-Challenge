import torch
from torch.utils.data import Dataset
import pickle
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import transform, Proj
import random

import sys
import datetime
import time

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# helper func to extract a window
from rasterio.windows import Window

def convert_from_EPSG4326(xy, dataset):
    """given a list of (lat, lon) encoded in EPSG:4326, returns a list of (x, y) encoded in dataset.crs"""
    inproj, outproj = Proj('EPSG:4326'), dataset.crs

    return list(zip(*transform(inproj, outproj, *zip(*xy))))

def convert_to_EPSG4326(xy, dataset):
    """given a list of (x, y) encoded in dataset.crs, returns a list of (x, y) encoded in EPSG4326"""

    inproj, outproj = dataset.crs, Proj('EPSG:4326')

    return list(zip(*transform(inproj, outproj, *zip(*xy))))

def extract_locs(dataset_path, xy=True, csv_filepath=None, size=(9, 9), stride=(9, 9)):
    
    validlist = []
    latlon = []
    # linear scan of image to extract valid locations
    with rasterio.open(dataset_path, mode="r") as dataset:
        for y in range(0, dataset.height, stride[1]):
            xbounds = [dataset.width, 0]
            for x in range(0, dataset.width // 2, stride[0]):
                window = Window(x - size[0] // 2, y - size[1] // 2, size[0], size[1])
                clip = dataset.read(window=window)
                clip[clip==-999.] = np.nan
                if (not np.isnan(np.sum(clip))) and (clip.shape[1:3] == size):
                    xbounds[0] = x
                    break
            for x in range(dataset.width, dataset.width // 2, -stride[0]):
                window = Window(x - size[0] // 2, y - size[1] // 2, size[0], size[1])
                clip = dataset.read(window=window)
                clip[clip==-999.] = np.nan
                if (not np.isnan(np.sum(clip))) and (clip.shape[1:3] == size):
                    xbounds[1] = x
                    break

            validlist.extend([(px, y) for px in range(xbounds[0], xbounds[1], stride[0])])

        coordlist = [dataset.xy(y, x) for x, y in validlist]
        latlon = convert_to_EPSG4326(coordlist, dataset)
    if xy:
        outlist = [(lat, lon, x, y) for (lat, lon), (x, y) in zip(latlon, validlist)]
        df = pd.DataFrame(outlist, columns=["LATITUDE", "LONGITUDE", "x", "y"])
    else: 
        df = pd.DataFrame(latlon, columns=["LATITUDE", "LONGITUDE"])
    if csv_filepath is not None:
        df.to_csv(csv_filepath, index=False)
    
    return df

def predict(model, dataloader, device=None):
    # find device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    outputs = {"LATITUDE": [],
               "LONGITUDE": [],
               "Prediction": [],
               "Probability": []
               }
    # setting model on eval mode
    t0 = time.time() # keep track of time taken

    print("Predicting :OOO ...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            lat, lon, img = batch["LATITUDE"], batch["LONGITUDE"], batch["image"].to(device)

            predictions = model(img)

            prob, pred = torch.max(predictions, 1)

            outputs["LATITUDE"].extend(lat.cpu().numpy().tolist())
            outputs["LONGITUDE"].extend(lon.cpu().numpy().tolist())
            outputs["Prediction"].extend(pred.cpu().numpy().tolist())
            outputs["Probability"].extend(prob.cpu().numpy().tolist())

            if not i == 0:
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                sys.stdout.write("\r" + '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i, len(dataloader), elapsed))

    return outputs

class PredictorDataset(Dataset):
    '''Creates a dataset from an input rasterio dataset and a series of valid points,
    needs columns LONGITUDE and LATITUDE
    '''

    def __init__(self, dataset_path, cachedir, csv_filepath=None, xy=True, size=(9, 9), stride=(9, 9)):
        if csv_filepath is None:
            self.locations = extract_locs(dataset_path, xy=xy, size=size, stride=stride)
        else:
            try:
                self.locations = pd.read_csv(csv_filepath)
            except:
                self.locations = extract_locs(dataset_path, csv_filepath=csv_filepath, xy=xy, size=size, stride=stride)


        self.lbound = None

        #checking if csv is valid

        columns = self.locations.columns
        if "LATITUDE" not in columns or "LONGITUDE" not in columns:
            raise Exception("csv invalid, LATITUDE, LONGITUDE must be in a valid csv!")
        
        self.xy = xy

        self.cachedir = cachedir
        self.size = size
        self.dataset_dir = dataset_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        coords = {"LATITUDE": float(self.locations.loc[idx, "LATITUDE"]), "LONGITUDE": float(self.locations.loc[idx, "LONGITUDE"])}

        xy = (self.locations.loc[idx, "x"], self.locations.loc[idx, "y"])

        try:
            cubic = np.load("{}/{}_{}_{}_{}.npy".format(self.cachedir, self.size[0], coords["LATITUDE"], coords["LONGITUDE"], self.dataset_dir.split("/")[-1].split(".")[0]), allow_pickle=True)
        except FileNotFoundError:
            cubic = self.extract_cubic(self.dataset_dir, self.conv_coords(coords, reverse=True), xycoords=xy, size=self.size)
            np.save("{}/{}_{}_{}_{}.npy".format(self.cachedir, self.size[0], coords["LATITUDE"], coords["LONGITUDE"], self.dataset_dir.split("/")[-1].split(".")[0]), cubic)
        
        cubic = np.nan_to_num(cubic)

        if self.lbound is not None:
            cubic = self.normOP(cubic)

        cubic = torch.tensor(cubic, dtype=torch.float32)
        
        sample = {"LATITUDE": coords["LATITUDE"], "LONGITUDE": coords["LONGITUDE"], "image": cubic}

        return sample
    
    def __len__(self):
        return len(self.locations)

    def extract_cubic(self, dataset_dir, coords, xycoords=None, size=(9, 9)):
        '''takes in a dataset dir, a tuple of coord (lat, long), size of the extracted cubics, 
        and returns coords_length * size1 * size2 * n_bands'''
        if xycoords:
            with rasterio.open(dataset_dir) as dataset:
                window = Window(xycoords[0] - size[0] // 2, xycoords[1] - size[1] // 2, size[0], size[1])
                clip = dataset.read(window=window)
                output = np.transpose(clip, (1, 2, 0))
            return np.expand_dims(output, axis=0)

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

    def conv_coords(self, coords, reverse=False):
        '''converts (lat, lon) to coords used by this class'''
        if reverse:
            return coords["LATITUDE"], coords["LONGITUDE"]
        return {"LATITUDE": coords[0], "LONGITUDE": coords[1]}

def createImage(dataset_path, pred_dict, interest_idx, size=(15, 15)):
    with rasterio.open(dataset_path) as dataset:
        image = np.zeros((dataset.height, dataset.width), dtype=np.int16)

    latlon = convert_from_EPSG4326(list(zip(pred_dict["LATITUDE"], pred_dict["LONGITUDE"])), dataset)
    # not actually latlon, just coords in dataset native projection

    for (lat, lon), pred, prob in zip(latlon, pred_dict["Prediction"], pred_dict["Probability"]):
        if pred == interest_idx:
            y, x = dataset.index(lon, lat)
            
            image[(y - size[1] // 2):(y + size[1] // 2), (x - size[0] // 2):(x + size[0] // 2)] = int(prob*255)

    return image

import matplotlib.pyplot as plt
import matplotlib

def plotimg(dataset_path, overlay):
    with rasterio.open(dataset_path) as dataset:
        fig = plt.figure()
        plt.subplot(1, 1, 1)
        plt.imshow(np.transpose(dataset.read((1, 2, 3)), (1, 2, 0))[...,::-1].copy())
        plt.imshow(overlay, cmap="cividis", alpha=0.5)
        plt.show()