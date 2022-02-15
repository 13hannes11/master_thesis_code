# Annotation Preprocessing

This directory contains code to extract image metadata from the database. In the first step the metadata is converted to csv files. The second step then loads the metadata and creates small tiles out of all images that contain eggs. The corresponding information about these patches is stored in a json file which cna be read by [Focus Annotator
](https://github.com/13hannes11/focus_annotator).


## Environment Variables

To run the preprocessing you need to create a `.env` file or set the corresponding environmental variables directly. The content of the file should be all of the following:

For step 0, fetching data from the database you need:

```
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_NAME=
```

For step 1, cropping and extracting images with eggs you need:

```
IMG_SIZE=75
ROOT_IN="in"
```

The actual code can be either run in a docker-container, for that you can run `docker-compose up` inside the this directory. Make sure you edit the mount in the docker-compose to your directories:

```yaml
    volumes:
      - <path to your output directory>:/usr/src/app/out:z
      - <path to your input directory>:/usr/src/app/in:z
```


Alternatively, you can manually run these python steps:

```
python 0_fetch_from_database.py
```

and

```
python 1_splitting_into_patches.py
```


