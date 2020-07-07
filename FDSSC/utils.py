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

def convert_to_EPSG4326(xy, dataset):
    """given a list of (x, y) encoded in dataset.crs, returns a list of (x, y) encoded in EPSG4326"""
    from pyproj import transform, Proj
    inproj, outproj = dataset.crs, Proj('EPSG:4326')
    
    return list(zip(*transform(inproj, outproj, *zip(*xy))))


# training model

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys

def train(model, dataloader, epochs=1, loss_fn=None, optimizer=None, lr=0.0001, callbacks=[], device=None):
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

    for callback in callbacks:
        callback()

    return loss_vals

def validate_callback(model, dataloader, loss_fn=None, device=None):
    if not loss_fn:
        loss_fn = nn.CrossEntropyLoss()
    
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def validate():
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

            # calculating validation loss
            tmp_eval_accuracy = loss_fn(outputs, label)

            eval_accuracy += tmp_eval_accuracy
            # Track the number of batches
            nb_eval_steps += 1

        print("  Loss: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    return validate

