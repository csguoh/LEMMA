# LEMMA
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2307.09749.pdf)

An official Pytorch implement of the paper "Towards Robust Scene Text Image Super-resolution via Explicit Location Enhancement" (IJCAI2023).

Authors: *Hang Guo, Tao Dai, Guanghao Meng, and Shu-Tao Xia*

This work proposes the Location Enhanced Multi-ModAl network (LEMMA) to address the challenges posed by complex backgrounds in scene text images with explicit positional enhancement. The architecture of LEMMA is as follows.

![LEMMA-pipeline](https://github.com/csguoh/LEMMA/blob/main/assets/LEMMA-pipeline.png)


## Pre-trained Model

As the previous code is a bit of a mess, we re-organize the code and retrain our LEMMA.  The performance of this re-trained model is as follows (better performance than that reported in the paper).

| Text Recognizer |  Easy  | Medium |  Hard  | avgAcc |
| :-------------: | :----: | :----: | :----: | :----: |
|      CRNN       | 64.98% | 59.89% | 43.48% | 56.73% |
|      MORAN      | 76.90% | 64.28% | 46.84% | 63.60% |
|      ASTER      | 81.53% | 67.40% | 48.85% | 66.93% |

One can download this model using this [link](https://drive.google.com/file/d/1iuVc0fh5rQAT2Ep5KgyV1GnsXPkdI4V0/view?usp=share_link) which contains the parameters of both the super-resolution brach and guidance generation branch.

The log file of training is also available with this [link](https://drive.google.com/file/d/1xtNnJ3gUXO1FSaebn2_rCKlcOfNBZR6o/view?usp=share_link).



## Prepare Datasets

In this work, we use STISR datasets TextZoom and four STR benchmarks, i.e., ICDAR2015, CUTE80, SVT and SVTP for model comparison. All the datasets are `lmdb` format.  One can download these datasets from the this [link](https://drive.google.com/drive/folders/1uqr8WIEM2xRs-K6I9KxtOdjcSoDWqJNJ?usp=share_link) we have prepared for you. And please do not forget to accustom your own dataset path in `./comfig.yaml` ,  such as the parameter `train_data_dir` and `val_data_dir`.



## Text Recognizers

Following previous STISR works, we also use [CRNN](https://github.com/meijieru/crnn.pytorch), [MORAN](https://github.com/Canjie-Luo/MORAN_v2  ) and [ASTER](https://github.com/ayumiymk/aster.pytorch) as the downstream text recognizer.  

Moreover, the code  also supports some new text recognizers, such as [ABINet](https://github.com/FangShancheng/ABINet), [MATRN](https://github.com/byeonghu-na/MATRN) and [PARSeq](https://github.com/baudm/parseq).  You can find the detailed comparison using these three new text recognizers in the supplementary material we provided and can also test LEMMA with these recognizers by modifying the command (e.g., `--test_model='ABINet'`). Please download these pre-trained text recognition models from the corresponding repositories we have provided above.

You also need to modify the text recognizer model path in the  `./config.yaml` file. Moreover, we employ the text focus loss proposed by STT during model training, since this text focus loss uses a pre-trained transformer based text recognizer,  please download this recognition model [here](https://drive.google.com/file/d/1HRpzveBbnJPQn3-k_y2Y1YY4PcraWOFP/view?usp=drive_link) and also accustom the ckpt path.



## How to Run?

We have set some default hype-parameters in the `config.yaml` and `main.py`, so you can directly implement training and testing after you modify the path of datasets and pre-trained model.  

### Training

```
python main.py
```

### Testing

```
python main.py --test
```

**NOTE:**  You can also auccstom other hype-parameters in the `config.yaml` and `main.py` file, such as the `n_gpu`.



## Main Results

### Quantitative Comparison

![quantitative-comparison](https://github.com/csguoh/LEMMA/blob/main/assets/quantitative-comparison.png)



### Qualitative Comparison

![qualitative-comparison](https://github.com/csguoh/LEMMA/blob/main/assets/qualitative-comparison.png)



## Citation

If you find our work helpful, please consider citing us.

```
@article{guo2023towards,
  title={Towards Robust Scene Text Image Super-resolution via Explicit Location Enhancement},
  author={Guo, Hang and Dai, Tao and Meng, Guanghao and Xia, Shu-Tao},
  journal={arXiv preprint arXiv:2307.09749},
  year={2023}
}
```



## Acknowledgement

The code of this work is based on [TBSRN](https://github.com/FudanVI/FudanOCR/tree/main/scene-text-telescope), [TATT](https://github.com/mjq11302010044/TATT), and [C3-STISR](https://github.com/JingyeChen/C3-STISR). Thanks for your contributions.
