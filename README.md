# `dimensions`

Estimating the instrinsic dimensionality of image datasets

Code for:  *The Intrinsic Dimensionaity of Images and Its Impact On Learning* - Phillip Pope and Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein (ICLR 2021, spotlight)
* https://openreview.net/forum?id=XJk19XzGq2J


![Basenjis of Varying dimensionality](https://github.com/ppope/dimensions/blob/main/extras/basenjis.png?raw=true)


### Environment

This code was developed in the following environment
```
conda create dimensions python=3.6 jupyter matplotlib scikit-learn pytorch==1.5.0 torchvision cudatoolkit=10.2 -c pytorch
```

To generate new data of controlled dimensionality with GANs, you must install:
```
pip install pytorch-pretrained-biggan
```

To use the `shortest-path` method (Granata and Carnevale 2016) you must also compile the fast graph shortest path code `gsp` (written by Jake VdP + Sci-Kit Learn)

```
cd estimators/gsp
python setup.py install
```

### Generate data of controlled dimensionality

```
python generate_data/gen_images.py \
  --num_samples 1000 \
  --class_name basenji \
  --latent_dim 16 \
  --batch_size 100 \
  --save_dir samples/basenji_16
```


### Estimate dimension of generated samples

To run the MLE (Levina and Bickel) estimator on the synthetic GAN data generated above:
```
python main.py \
    --estimator mle \
    --k1 25 \
    --single-k \
    --eval-every-k \
    --average-inverse \
    --dset  samples/basenji_16 \
    --max_num_samples 1000 \
    --save-path results/basenji_16.json
```

Use `--estimators` to try different estimators


### Citation

If you find our paper or code useful, please cite our paper:
```
@inproceedings{DBLP:conf/iclr/PopeZAGG21,
  author    = {Phillip Pope and
               Chen Zhu and
               Ahmed Abdelkader and
               Micah Goldblum and
               Tom Goldstein},
  title     = {The Intrinsic Dimension of Images and Its Impact on Learning},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=XJk19XzGq2J},
  timestamp = {Wed, 23 Jun 2021 17:36:39 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/PopeZAGG21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Acknowledgements

We gratefully acknowledge use of the following codebases when developing our dimensionality estimators:
* [human-analysis/intrinsic-dimensionality](https://github.com/human-analysis/intrinsic-dimensionality)
* [stat-ml/GeoMLE](https://github.com/stat-ml/GeoMLE)
* [dgranata/Intrinsic-Dimension](https://github.com/dgranata/Intrinsic-Dimension)
* [jmmanley/two-nn-dimensionality-estimator](https://github.com/jmmanley/two-nn-dimensionality-estimator)
* [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [huggingface/pytorch-pretrained-BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN)


We also thank Prof. Vishnu Boddeti for clarifying comments on the graph-distance estimator.


### Disclaimer

This code released *as is*. We will do our best to address questions/bugs, but cannot guarantee support.
