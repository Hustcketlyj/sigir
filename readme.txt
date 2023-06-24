run the following to setup environment for clean.py (generate augmented sets)

!pip install datasets evaluate transformers[sentencepiece]
!pip install accelerate
!apt install git-lfs
from huggingface_hub import login
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
login('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')
!pip install jsonlines
from datasets import load_dataset
import json
import secrets
!pip install spacy
import spacy
nlp=spacy.load('en_core_web_sm')
import spacy
import jsonlines
from transformers import DataCollatorForLanguageModeling
import collections
import numpy as np
from transformers import TrainingArguments
from transformers import Trainer
from transformers import default_data_collator
import math
import spacy
from transformers import pipeline
nlp=spacy.load('en_core_web_sm')

install the following packages for allen.py (evaluate performance)

absl-py                      1.4.0
aiohttp                      3.8.4
aiosignal                    1.3.1
astunparse                   1.6.3
async-timeout                4.0.2
attrs                        23.1.0
blis                         0.7.9
cachetools                   5.3.0
catalogue                    2.0.8
certifi                      2022.12.7
charset-normalizer           3.1.0
click                        8.1.3
cmake                        3.26.3
confection                   0.0.4
cymem                        2.0.7
datasets                     2.11.0
dill                         0.3.6
en-core-web-sm               3.5.0
evaluate                     0.4.0
filelock                     3.12.0
flatbuffers                  23.5.9
frozenlist                   1.3.3
fsspec                       2023.4.0
gast                         0.4.0
google-auth                  2.18.0
google-auth-oauthlib         1.0.0
google-pasta                 0.2.0
grpcio                       1.54.2
h5py                         3.8.0
huggingface-hub              0.14.1
idna                         3.4
importlib-metadata           6.6.0
jax                          0.4.10
Jinja2                       3.1.2
joblib                       1.2.0
jsonlines                    3.1.0
keras                        2.12.0
langcodes                    3.3.0
libclang                     16.0.0
lit                          16.0.2
Markdown                     3.4.3
MarkupSafe                   2.1.2
ml-dtypes                    0.1.0
mpmath                       1.3.0
multidict                    6.0.4
multiprocess                 0.70.14
murmurhash                   1.0.9
networkx                     3.1
numpy                        1.23.5
nvidia-cublas-cu11           11.10.3.66
nvidia-cuda-cupti-cu11       11.7.101
nvidia-cuda-nvrtc-cu11       11.7.99
nvidia-cuda-runtime-cu11     11.7.99
nvidia-cudnn-cu11            8.5.0.96
nvidia-cufft-cu11            10.9.0.58
nvidia-curand-cu11           10.2.10.91
nvidia-cusolver-cu11         11.4.0.1
nvidia-cusparse-cu11         11.7.4.91
nvidia-nccl-cu11             2.14.3
nvidia-nvtx-cu11             11.7.91
oauthlib                     3.2.2
opt-einsum                   3.3.0
packaging                    23.1
pandas                       2.0.1
pathy                        0.10.1
Pillow                       9.5.0
pip                          23.0.1
preshed                      3.0.8
protobuf                     4.23.0
pyarrow                      11.0.0
pyasn1                       0.5.0
pyasn1-modules               0.3.0
pydantic                     1.10.7
python-dateutil              2.8.2
pytz                         2023.3
PyYAML                       6.0
regex                        2023.3.23
requests                     2.29.0
requests-oauthlib            1.3.1
responses                    0.18.0
rsa                          4.9
scikit-learn                 1.2.2
scipy                        1.10.1
sentencepiece                0.1.98
setuptools                   59.5.0
six                          1.16.0
smart-open                   6.3.0
spacy                        3.5.2
spacy-legacy                 3.0.12
spacy-loggers                1.0.4
srsly                        2.4.6
sympy                        1.11.1
tensorboard                  2.12.3
tensorboard-data-server      0.7.0
tensorflow                   2.12.0
tensorflow-estimator         2.12.0
tensorflow-io-gcs-filesystem 0.32.0
termcolor                    2.3.0
thinc                        8.1.9
threadpoolctl                3.1.0
tokenizers                   0.13.3
torch                        1.10.1+cu111
torchaudio                   0.10.1+cu111
torchvision                  0.11.2+cu111
tqdm                         4.65.0
transformers                 4.28.1
triton                       2.0.0
typer                        0.7.0
typing_extensions            4.5.0
tzdata                       2023.3
urllib3                      1.26.15
wasabi                       1.1.1
Werkzeug                     2.3.4
wheel                        0.38.4
wrapt                        1.14.1
xxhash                       3.2.0
yarl                         1.9.2
zipp                         3.15.0

#############################################
run clean.py first, then allen.py