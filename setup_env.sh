#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
APP_NAME=${PWD##*/}
echo "Rebuilding venv for ${APP_NAME}..."
PY_EXE=python3
rm -fr .venv/
$PY_EXE -m venv .venv
source .venv/bin/activate
.venv/bin/$PY_EXE -m pip install --upgrade pip
.venv/bin/$PY_EXE -m pip install -r requirements.txt
