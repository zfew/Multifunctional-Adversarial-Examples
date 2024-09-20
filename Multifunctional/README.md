## Multifunctional Adversarial Examples: A Novel Mechanism for Authenticatable Privacy Protection of Images

This is a Pytorch implementation of "Multifunctional Adversarial Examples: A Novel Mechanism for Authenticatable Privacy Protection of Images".

### Usage

#### Train

Prepare the cover images for training, validation, testing in separate folders first. To train a new model, one may use `main.py`. An example is:

```shell
python main.py new --train-data-dir *cover path for training* --valid-data-dir *cover path for validation* --cover-data-dir *cover save folder* --stego-data-dir *Adversarial Examples save folder* --run-folder *experiment folder* --title *experiment name* --use-vgg
```





