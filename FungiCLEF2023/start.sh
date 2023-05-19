#!/bin/bash

METADATA_CSV='http://ptak.felk.cvut.cz/plants/DanishFungiDataset/FungiCLEF2023_val_metadata_PRODUCTION.csv'
DATA_ROOT_PATH='/Data'
python ./predict.py \
    --input-file $METADATA_CSV \
    --data-root-path $DATA_ROOT_PATH \
    --output-file user_submission.csv && \
python ./evaluate.py \
    --test-annotation-file $METADATA_CSV \
    --user-submission-file user_submission.csv
