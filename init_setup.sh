echo [$(date)]: "Start"

echo [$(date)]: "creating env with python 3.8 version"

conda create -p ./env python=3.8 -y

echo [$(date)]: "activating the envinorment"

source activate ./env

echo [$(date)]: "installing the dev requirements"

pip install -r requirements.txt

echo [$(date)]: "END"