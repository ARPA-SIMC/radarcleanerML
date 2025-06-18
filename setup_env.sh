#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
APP_NAME=${PWD##*/}

OPENDATA_BASEURL="https://dati-simc.arpae.it/opendata/radarcleanerML"
echo "Downloading dataset from $OPENDATA_BASEURL"
wget -nv --directory-prefix=data --no-parent --reject "index.html*" -r -nH --cut-dirs=2 $OPENDATA_BASEURL/

echo "Rebuilding venv for ${APP_NAME}..."
PY_EXE=python3
rm -fr .venv/
$PY_EXE -m  venv --system-site-packages .venv
source .venv/bin/activate
.venv/bin/$PY_EXE -m pip install --upgrade pip
.venv/bin/$PY_EXE -m pip install -r requirements.txt
if ! python -c 'import importlib.metadata; importlib.metadata.version("simcradarlib")' 2>/dev/null
then
    pip install "simcradarlib @ git+https://github.com/ARPA-SIMC/simcradarlib@v0.6-1"
fi

./scripts/create_dset.sh
