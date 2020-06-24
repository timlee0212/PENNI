# PENNI
This repo provides the code of [PENNI: Pruned Kernel Sharing for Efficient CNN Inference](https://arxiv.org/abs/2005.07133).

Install Requirement Packages:

    pip install -r requirements.txt

If you find this work is helpful, cite with:

    @inproceedings{li2020penni,
    title={PENNI: Pruned Kernel Sharing for Efficient CNN Inference},
    author={Li, Shiyu and Hanson, Edward and Li, Hai and Chen, Yiran},
    booktitle={International Conference on Machine Learning},
    year={2020}
    }

## Acknowledgement

The resnet-56 implementation is from: [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)

We count the FLOPs number with the modified version of [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)

The ImageNet training script is derived from [apex](https://github.com/NVIDIA/apex)
