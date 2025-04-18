# for sagemaker setup, we avoid messing with system-level installations
apt-get update -y
#apt-get install tmux -y
#apt-get install python3-dev -y

apt-get install libaio-dev libopenmpi-dev python3-mpi4py vim -y

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv p2l_env --python 3.10

source p2l_env/bin/activate

#uv pip install wheel packaging

#uv pip install -r train_requirements.txt
#uv pip install flash-attn==2.5.9.post1 --no-build-isolation

uv pip install fastapi uvicorn mpi4py
python -m ipykernel install --user --name=p2l_env --display-name "Python (p2l_env)"
uv pip install -e .
