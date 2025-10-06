
Jax doens't support Windows
Powershell
```bash
wsl --install -d Ubuntu-22.04
```


connect to Ubuntu in vscode or cursor

check your system
```shell
nvidia-smi
```
->
```
NVIDIA-SMI 570.133.07             Driver Version: 572.83         CUDA Version: 12.8  
```

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
```

```bash
pip install mujoco-mjx
pip install matplotlib
pip install -U "jax[cuda12]"
# pip install jaxlib
```


## Tips

- GPU watch on Linux:
    ```
    watch -n 0.5 nvidia-smi
    ```