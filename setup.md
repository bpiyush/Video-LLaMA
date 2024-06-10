

* Create environment.

```sh
# conda env create -f environment.yml
# conda activate videollama

conda create -n videollama python=3.9 -y
conda activate videollama

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements_specific.txt 
```

* Download checkpoints

```python
from huggingface_hub import snapshot_download
repo_id = "DAMO-NLP-SG/Video-LLaMA-2-7B-Pretrained"
snapshot_download(repo_id=repo_id)
```

(Move the checkpoints from `vgg-download` to `vggdev21`.)