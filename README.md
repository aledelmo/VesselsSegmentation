# VesselsSegmentation
Semi-automatic segmentation of veins and arteries using a deep learning approach

## Docker

docker build -t test_docker_vessels . && docker run -v /Users/imag2/Desktop/VesselsSegmentation/test_dataset:/test_docker --name DeepVessel test_docker_vessels 

## Other repositories

This software is part of the IMAG2 framework. It can be complemented by:
* [3DSlicer Plug-ins]: segmentation and diffusion extension for 3DSlicer
* [PQL]: the first ever method for the segmentation of pelvic tractograms.
* [Tractography Metrics]: tool for the analysis of fiber bundles.
* [IMAG2 Utilities]: collection of various scripts.
* [IMAG2 Website]: the completely redesigned team website (<http://www.imag2.org>)

 License
----

Apache License 2.0

[//]: #
   [3DSlicer Plug-ins]: <https://github.com/aledelmo/3DSlicer_Plugins>
   [PQL]: <https://github.com/aledelmo/PQL>
   [Tractography Metrics]: <https://github.com/aledelmo/TractographyMetrics>
   [Vessel Segmentation]: <https://github.com/aledelmo/VesselsSegmentation>
   [IMAG2 Utilities]: <https://github.com/aledelmo/IMAG2_Utilities>
   [IMAG2 Website]: <https://github.com/aledelmo/IMAG2_Website>