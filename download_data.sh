#!/bin/bash
# Run from a machine/compute session with outbound web access (not inside a GPU job).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/raw
curl --fail --location --retry 3 --continue-at - --output data/raw/OmicsExpressionProteinCodingGenesTPMLogp1.csv 'https://ndownloader.figshare.com/files/51065489'
curl --fail --location --retry 3 --continue-at - --output data/raw/Model.csv 'https://ndownloader.figshare.com/files/51065297'
curl --fail --location --retry 3 --continue-at - --output data/raw/GDSC2_fitted_dose_response_27Oct23.xlsx 'https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC2_fitted_dose_response_27Oct23.xlsx'
ls -lh data/raw/*
