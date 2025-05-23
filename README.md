Papers: 

Liang, Yuting, and Reza Samavi. "Advanced defensive distillation with ensemble voting and noisy logits." Applied Intelligence 53.3 (2023): 3069-3094.

Yuting Liang and Reza Samavi. Towards Robust Deep Learning with Ensemble Networks and Noisy Layers. AAAI Workshop on RSEML, 2021 (https://arxiv.org/abs/2007.01507)



# nn_robust_ensemble

## Getting Started
A large part of the code is based on Carlini's nn_robust_attacks project https://github.com/carlini/nn_robust_attacks. The functions written specifically for this project are preceded by comments to briefly describe what they are. Comments also appear in important steps in the code.

### Prerequisites
The following python modules are required to run this code:
- Numpy (general array manipulations and utilities)
- Tensorflow 
- Keras (neural network models)
- Matplotlib (graphing utilities)
```
pip install matplotlib
```
### Training the ensemble
To train the ensemble of networks, run the script _train_models_main.py_.

### Running the tests
Both tests for transferability (single network attack) as well as superimposition attacks are in the script _test_models_main.py_. In the script, specifiy the dataset (MNIST or CIFAR) and one of the following models:
- MNISTModel (simple MNIST ensemble)
- MNISTDPModel (MNIST ensemble with noisy inputs)
- CIFARModel (simple CIFAR ensemble)
- CIFARDPModel (CIFAR ensemble with noisy inputs)

### Processing the results (optional)
The script _process_results_main.py_ can be used to process the raw results from the tests. It will generate statistics from the test result files generated by _test_models_main.py_.

## Cite
If you find the content useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{Liang2023,
    author={Liang, Yuting
    and Samavi, Reza},
    title={Advanced defensive distillation with ensemble voting and noisy logits},
    journal={Applied Intelligence},
    year={2023},
    month={Feb},
    day={01},
    volume={53},
    number={3},
    pages={3069-3094},
    abstract={In this paper we provide an approach for deep neural networks that protects against adversarial examples in image classification-type problems. blackUnlike adversarial training, our approach is independent to the obtained adversarial examples through min-max optimization. The approach relies on the defensive distillation mechanism. This defence mechanism, while very successful at the time, was defeated in less than a year due to a major intrinsic vulnerability: the availability of the neural network's logit layer to the attacker. We overcome this vulnerability and enhance defensive distillation by two mechanisms: 1) a mechanism to hide the logit layer (noisy logit) which increases robustness at the expense of accuracy, and, 2) a mechanism that improves accuracy but does not always increase robustness (ensemble network). We show that by combining the two mechanisms and incorporating a voting method, we can provide protection against adversarial examples while retaining accuracy. We formulate potential attacks on our approach with different threat models. The experimental results demonstrate the effectiveness of our approach. We also provide a robustness guarantee along with an interpretation for the guarantee.},
    issn={1573-7497},
    doi={10.1007/s10489-022-03495-3},
    url={https://doi.org/10.1007/s10489-022-03495-3}
}
```

and

```bibtex
@misc{liang2021robustdeeplearningensemble,
    title={Towards Robust Deep Learning with Ensemble Networks and Noisy Layers}, 
    author={Yuting Liang and Reza Samavi},
    year={2021},
    eprint={2007.01507},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2007.01507}, 
}
```

