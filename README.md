# Patch-Based Vessels Segmentation with Transfer Learning

*Deprecation Note: this implementation dates from 2018. The framework is outdated. For more recent AI semantic segmentation
algorithms please check [APMRI-DNN](https://github.com/aledelmo/APMRI-DNN)*.

Vascular tree segmentation with transfer learning and path reconstruction.

![vessels](https://i.imgur.com/Xo2WBqx.jpg)

The algorithm is divided in two steps:
1.  Path reconstruction from landmarks
2.  Semantic segmentation using transfer learning

A set of landmarks along the vessels is provided by the user. Iteratively, we build a tree-like structure assigning each
leaf to the parent node that minimize the following objective function:

![f_phi](https://i.imgur.com/WLrozlu.jpg)

Minimizing f means that the path should be formed by points as close as possible, forming a line as straight as possible,
and whose spatial context is homogeneous in terms of intensity.

Patches are centered on the vessels branches in each slice of the image volume. We use the Bresenham's line algorithm [1]
to reconstruct the vascular tree.

Vessels regions are jointly classified into veins or arteries. We use as base network the VGG-16 [2] pre-trained on the 
ImageNet dataset. A specialized convolutional layer is inserted after the last convolutional layer of each stage. 
Feature maps in the concatenated layers are linearly combined through a final convolutional layer.

## Usage

```shell
$ docker build --build-arg https_proxy=https://<address>:<port> -f Dockerfile --rm --tag vessels_dl .
$ docker run -u $(id -u) --gpus all -it -v $(PWD)/test_dataset:/test_docker --env HTTPS_PROXY=https://<address>:<port> --rm --name deep_vessels vessels_dl:latest
```

## System Requirements

Project developed using [caffe](https://github.com/BVLC/caffe).

[Docker](https://www.docker.com) CE 20.10.3 with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) for GPU acceleration

## Contacts

For any inquiries please contact: 
[Alessandro Delmonte](https://aledelmo.github.io) @ [alessandro.delmonte@institutimagine.org](mailto:alessandro.delmonte@institutimagine.org)

## Reference

If you use this work please cite:

```bibtex
@inproceedings{https://hal.archives-ouvertes.fr/hal-02287946,
  author = {Virzi, Alessio and Gori, Pietro and Muller, C{\'e}cile and Mille, Eva and Peyrot, Quoc and Berteloot, Laureline and Boddaert, Nathalie and Sarnacki, Sabine and Bloch, Isabelle},
  title = {Segmentation of pelvic vessels in pediatric MRI using a patch-based deep learning approach},
  journal = {PIPPI MICCAI Workshop},
  address = {Granada, Spain},
  volume = {LNCS 11076},
  pages = {97-106},
  year = {2018},
}
```
*[1] Bresenham, J. E. (1965). "Algorithm for computer control of a digital plotter". IBM Systems Journal. 4(1): 25–30. doi:10.1147/sj.41.0025.*

*[2] Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. CoRR abs/1409.1556 (2014)*

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for
details
