# helper functions
import datetime

def ceildiv(a, b):
    return - (- a // b)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def find_bbox(dataset):
    """given a dataset, will return the bounding box in original coords (tl, tr, br, bl)"""
    tl = dataset.transform * (0, 0)
    tr = dataset.transform * (dataset.width, 0)
    bl = dataset.transform * (0, dataset.height)
    br = dataset.transform * (dataset.width, dataset.height)

    return tl, tr, br, bl

# Helper Variables

wantedids = ["Pb", "Mn", "U", "Au", "Cu", "Fe", "Ag", "Ni"]

# helper funcs to process csvs

def get_id2idx(df, id_column):
    '''get id to index representation given a dataframe and an id column name'''
    df_dict = df.to_dict(orient="list")
    no_rep = list(set(df_dict[id_column]))
    if "NONE" in no_rep:
        no_rep.remove("NONE")
        outdict = {"NONE": 0}
        for i, rep in enumerate(no_rep):
            outdict[rep] = i + 1
        return outdict

    return {ident: idx for idx, ident in enumerate(no_rep)}

def create_idxlist(df, id_column, id2idx):
    '''create a list of "labels" from dataframe, id column name, and id2idx dictionary'''
    df_dict = df.to_dict(orient="list")

    return [id2idx[each] for each in df_dict[id_column]]

def splitData(df, train=0.8, val=0.1):
    size = df.shape[0]
    traindf = df.iloc[0:int(train*size)]
    valdf = df.iloc[int(train*size):int((train + val) * size)]
    testdf = df.iloc[int((train + val) * size):]

    return traindf, valdf, testdf

def take_classes(df, id_column, ids):
    '''takes a list of ids and discards all ids that aren't in the list'''

    return df.loc[df[id_column].isin(ids)]

def mask_classes(df, id_column, ids):
    '''takes a list of ids and makes the id's not in the list a null value'''
    
    df.loc[~df[id_column].isin(ids), id_column] = "NONE"
    return df

def freqdict(df, id_column):
    df_dict = df.to_dict(orient="list")
    no_rep = list(set(df_dict[id_column]))
    
    outdict = {}

    for each in no_rep:
        outdict[each] = df_dict[id_column].count(each)

    return outdict

# import transform library
# stuff to transform images and stuff
from pyproj import transform, Proj

def convert_to_EPSG4326(xy, dataset):
    """given a list of (x, y) encoded in dataset.crs, returns a list of (x, y) encoded in EPSG4326"""

    inproj, outproj = dataset.crs, Proj('EPSG:4326')

    return list(zip(*transform(inproj, outproj, *zip(*xy))))

def convert_from_EPSG4326(xy, dataset):
    """given a list of (x, y) encoded in EPSG:4326, returns a list of (x, y) encoded in dataset.crs"""
    inproj, outproj = Proj('EPSG:4326'), dataset.crs

    return list(zip(*transform(inproj, outproj, *zip(*xy))))

def findshape(dataset, crs='EPSG:4326'):
    """Outputs in latitude, longitude"""
    import rasterio
    import rasterio.features
    import rasterio.warp

    # Read the dataset's valid data mask as a ndarray.
    mask = dataset.dataset_mask()

    outlist = []
    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(
            mask, transform=dataset.transform):

        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        geom = rasterio.warp.transform_geom(
            dataset.crs, crs, geom, precision=6)
        
        outlist.append(geom)
    
    return outlist

def collate_images(dirlist):
    import rasterio
    for directory in dirlist:
        arrays = {}
        # Read metadata of first file
        with rasterio.open(directory[0]) as src0:
            metadata = src0.meta

        metadata.update(count=len(directory))
        with rasterio.open(("_".join(directory[0].split("/")[-1].split(".")[0].split("_")[1:-1]) + ".TIF"), "w", **metadata) as dest:
            for image in directory:
                with rasterio.open(image) as dset:
                    bnum = int(image.split(".")[0].split("_")[-1].replace("B", ""))
                    dest.write_band(bnum, dset.read(1))
    return "yeetus reetus"
    
def walkdir(datadir):
    import os

    dirlist = []

    for root, dirs, files in os.walk(datadir):
        for name in dirs:
            dirlist.append(root + "/" + name)

    dir2files = {}
    for name in dirlist:
        for root, dirs, files in os.walk(name):
            namelist = []
            for name in files:
                if "TIF" in name:
                    namelist.append(root + "/" + name)
            dir2files[root.split("/")[-1]] = namelist

    return dir2files

# extract images
from rasterio.windows import Window
import numpy as np
def extract_cubic(dataset, coords, size=(9, 9)):
    '''takes in a opened rasterio dataset, a list of tuples of coords [(lat, long),], size of the extracted cubics, 
    and returns coords_length * size1 * size2 * n_bands'''
    
    output = np.zeros((len(coords), *size, dataset.count))
    trans_coords = convert_from_EPSG4326(coords, dataset)
    
    for i, (lon, lat) in enumerate(trans_coords):
        py, px = dataset.index(lon, lat)
        window = Window(px - size[0] // 2, py - size[1] // 2, size[0], size[1])
        clip = dataset.read(window=window)
        output[i] = np.transpose(clip, (1, 2, 0))

    return output

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

# initializing model weights
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight, gain=2.)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

# training model

def train(model, dataloader, epochs=1, loss_fn=None, optimizer=None, lr=0.0001, callbacks=[], device=None, callback_args={}):
    # define default loss functions and optimizer
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss() #x is (batch_size, classes), targets are integers that correspond to the index of class (batch_size,)
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # find device
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting model to training mode
    model.train()

    loss_vals = [] # creating list to store loss values

    callback_outputs = []

    for epoch in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')

        t0 = time.time() # keep track of time taken
        running_loss = 0.0 # keep track of running loss
        
        # training model
        for i, data in enumerate(dataloader):
            inputs, label = data["image"].to(device), data["label"].to(device)

            optimizer.zero_grad()

            output = model(inputs)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            # keeping track of a running loss
            running_loss += loss.item()

            # report progress for each batch
            if not i == 0:
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                sys.stdout.write("\r" + '  Batch {:>5,}  of  {:>5,}.    Loss: {:.2f}     Elapsed: {:}.'.format(i, len(dataloader), loss.item(), elapsed))

        #calculating the average training loss
        avg_train_loss = running_loss / len(dataloader)
        loss_vals.append(avg_train_loss)
        #printing updates
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))
    
        callback_args["loss"] = avg_train_loss
        callback_args["epoch"] = epoch + 1

        callback_returns = []
        for callback in callbacks:
            callback_returns.append(callback(callback_args))
        callback_outputs.append(callback_returns)

    return loss_vals, callback_outputs

def validate_callback(model, dataloader, loss_fn=None, device=None):
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss()
    
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def validate(kwargs):
        print("Running Validation...")

        t0 = time.time()
        model.eval()

        eval_accuracy = 0
        nb_eval_steps = 0

        for batch in dataloader:
            inputs, label = batch["image"].to(device), batch["label"].to(device)

            with torch.no_grad():
                #forward pass without gradient calculations for speeeeeeeeed

                outputs = model(inputs)

            # Getting predicted value
                _, predicted = torch.max(outputs.data, 1)
            # calculating validation loss
            total = label.size(0)
            correct = (predicted == label).sum().item()

            tmp_eval_accuracy = correct / total

            eval_accuracy += tmp_eval_accuracy
            # Track the number of batches
            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(100 * eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        return eval_accuracy / nb_eval_steps
    return validate

def saving_checkpoints_callback(path2chckpt, model, optimizer, freq, epochs):

    def save(kwargs):
        epoch = kwargs["epoch"]
        loss = kwargs["loss"]
        if ((epoch - 1) % freq == 0 or epoch == epochs) and (epoch != 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, "{}{}epoch_{}loss.pt".format(path2chckpt, epoch, loss))
        
    return save
