# JCCM Signals Recognition

[![DOI:10.1007/978-3-319-76207-4_15](https://zenodo.org/badge/DOI/10.1049/ell2.13006.svg)](https://doi.org/10.1049/ell2.13006) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com)

>
>
>Liang J, Li X, Liang C, et al. JCCM: Joint conformer and CNN model for overlapping radio signals recognition[J]. Electronics Letters, 2023, 59(21): e13006.

## Poster

![img_0](./imgs/poster.jpg)

## Model Description

Joint Conformer and CNN model(JCCM) is proposed to separate and recognize the overlapping radio signals which are also unknown by the monitor node. JCCM utilizes the attention mechanism of conformer to encode signal spectrum into feature maps and decodes the feature maps into signal component proportions by perceiving the global and local features through convolutional neural networks (CNN). In addition, a signal preprocessing module is designed including MinPool and AvgPool layers for signal denoising .

![img_1](./imgs/git_1.png)

The model was introduced in the paper [JCCM: Joint conformer and CNN model for overlapping radio signals recognition](https://doi.org/10.1049/ell2.13006) .

## Results

![img_2](./imgs/git_2.png)

- the cosine similarity comparison of the above models for SNR between−45 dB and 15 dB. Compared with NMF, SVR and Lasso Regression, the use of deep neural network methods have better performance on recognition accuracy and restore the proportion of each signal component in the mixed signals. Our proposed model JCCM also shows more increasing performance in the low SNR (from−30 dB to−5dB) than MLP-3L and ResNet18. Moreover, applying SP module greatly improves the similarity performance due to the noise affect reduction on signal recognition.
- he principal signal classification accuracy of the above models with SNR between−45 dB and 15 dB. JCCM achieves significantly better results. The accuracy of JCCM is around 5% to 25% higher than ResNet18 when SNR is between−20 dB and 0 dB. When SNR is−25 dB, JCCM can maintain 30% classification accuracy even when other models fail, and it can reach 38% when using the SP module. This observation suggests that the enhancement on the uniformity of signal feature extraction by SP module has greatly facilitated the encoder and decoder in JCCM.

## Example

![img_3](./imgs/git_3.png)

An example of scaling preservation through the overlapping radio signals separation in SNR=0dB by using JCCM is shown in this figure.

## Requirements

- Python 3.9
- torch 1.12.1
- tqdm 4.66.1

## References

1. Gulati A, Qin J, Chiu C C, et al. Conformer: Convolution-augmented transformer for speech recognition[J]. arXiv preprint arXiv:2005.08100, 2020. [Github](https://github.com/sooftware/conformer) 



