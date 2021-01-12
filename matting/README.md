## Matting

Replaces any background with a video file using Bayes inference or neural network

`matting.cpp` - accepts original video, image with a trimap (see video version on how to construct one) and a keyframe. The last parameter is algorithm to use `mlp|bayes`. The result is a video file with alpha map

`matting.h` - header, that defines algorithm classes and also implements strategy pattern fir dynamic algo selection in main program. 

`trimap.cpp|.h` - some utilities for trimap processing

`mlp.cpp|bayess.cpp` - implementations of algorithms

`grid_search.cpp` - searching to best hyper-parameters for MLP, by producing a alphamaps

`alpha_merge.cpp` - uses alpha map video from `matting` to replace background with the new one



