# capgen


## Getting Started

  1. Clone the repo

    $ git clone https://github.com/tazzaoui/capgen.git && cd capgen

  2. Download Training Data

    $ cd scripts && ./get_data.sh

  3. Resize Training Images

    $ ./resize.sh

## Training the model 

Use `train.py` to train the model

`$ ./train.py`

An neural image caption generator based on Google's show and tell paper: (https://arxiv.org/pdf/1411.4555.pdf)

Dataset: Flickr8k Dataset available via the request form here:https://forms.illinois.edu/sec/1713398
