### VFX with OpenCV and C++

## [Chroma Keying](https://smirnov-am.github.io/chromakeying/) | [Video version](https://youtu.be/Q7X4agNKU3k) | [Demo](https://www.youtube.com/watch?v=Q7X4agNKU3k&t=397s)

Replaces green background in a video with provided picture. 

`chrome.cpp` - main file, accepts path to video, path to new background image and 2 parameters that tune the algorithm.

## Matting
Replaces any background with a video file using Bayes inference or neural network

`matting.cpp` - accepts original video, image with a trimap (see video version on how to construct one) and a keyframe. The last parameter is algorithm to use `mlp|bayes`

`matting.h` - header, that defines algorithm classes and also implements strategy pattern fir dynamic algo selection in main program. The result is a video file with alpha map

`trimap.cpp|.h` - some utilities for trimap processing

`mlp.cpp|bayess.cpp` - implementations of algorithms

`grid_search.cpp` - searching to best hyper-parameters for multilayer perceptron, by producing a alpha_maps

`alpha_merge.cpp` - uses alpha map video from `matting` to replace background with the new one



