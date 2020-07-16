def collate_images(dirlist):
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

# legacy
    datadir = "data"

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
            dir2files[root] = namelist

# legacy plotting
    fig, ax = plt.subplots(3, 4)
    fig2, ax2 = plt.subplots(3, 4)

    with rasterio.open("LC08_L1TP_098082_20200125_20200128_01_T1.TIF") as dataset:
        for i in range(0, 11):
            j = i // 4
            k = i % 4
            print(j, k, i)
            ax[j][k].imshow(dataset.read(i+1), cmap="Greens")

    for each in list(dir2files.values())[0]:
        bnum = int(each.split(".")[0].split("_")[-1].replace("B", ""))
        j = (bnum - 1) // 4
        k = (bnum - 1) % 4
        with rasterio.open(each) as dataset:
            ax2[j][k].imshow(dataset.read(1), cmap="Greens")


# legacy map display
    dir2name = utils.walkdir(datadir)

    m = folium.Map(location=[-31.743944369555788, 138.29644592694788], tiles='OpenStreetMap', zoom_start=8)


    tile = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri', name = 'Esri Satellite', overlay = False, 
            control = True, opacity=0.7).add_to(m)
    root = "data/stacked_images"
    for files in list(dir2name.values())[0]:
            dataset = rasterio.open(files)
            image_loc = folium.vector_layers.Polygon([(y, x) for x, y in utils.findshape(dataset)[0]["coordinates"][0]]).add_to(m)
    for i in OCC_df.index:
        folium.Marker([OCC_df.loc[i,'LATITUDE'], 
                        OCC_df.loc[i,'LONGITUDE']]).add_to(m)

    m