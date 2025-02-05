apt-get update -y
apt-get install tmux -y

apt-get install tmux libaio-dev libopenmpi-dev python3-mpi4py -y

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv p2l --python 3.10

source p2l/bin/activate

uv pip install wheel packaging

uv pip install -r requirements.txt
uv pip install flash-attn==2.5.9.post1 --no-build-isolation
