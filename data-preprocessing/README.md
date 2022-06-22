# Data Preprocessing

This directory contains code to convert data that was annotated using Focus Annotator from json to csv format. 

## What It Does

The script loops over all .json files in `IN_FOLDER` and splits them randomly according to the defined split environmental variables. The splits from each input file are concatenated into files called `train_metadata.csv`, `validation_metadata.csv` and `test_metadata.csv`.


## Environment Variables
 test
To run the preprocessing you need to create a .env file or set the corresponding environmental variables directly. And then simply run the python script.
The content of the file should be all of the following:

```
IN_FOLDER=in/
OUT_FOLDER=out/

TRAIN_SPLIT=0.7
VALIDATION_SPLIT=0.15
TEST_SPLIT=0.15
```

