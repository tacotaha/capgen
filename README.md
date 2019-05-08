# capgen
> A Neural Image Caption Generator


## Results

This model was trained on the MSCOCO train2014 dataset and obtains the following results

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr|
| ---| --- | --- | --- | --- | --- | ---| 
| 0.651 | 0.467 | 0.322 | 0.218 | 0.212 | 0.479 | 0.690|


## Getting Started

  1. Clone the repo

    $ git clone https://github.com/tazzaoui/capgen.git && cd capgen

  2. Download MS-COCO Training Data

    $ cd scripts && ./get_data.sh

  3. Resize Training Images

    $ ./resize.sh

## Training the model 

Use `train.py` to train the model

`$ ./train.py`




## Sources
Google's show and tell paper: (https://arxiv.org/pdf/1411.4555.pdf)

resnet paper: (https://arxiv.org/abs/1512.03385)

Dataset: (http://cocodataset.org/#home)
