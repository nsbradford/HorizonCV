# HorizonCV

#### Nicholas Bradford

Real-time horizon detection with computer vision in Python, for UAV applications. Algorithm source found in [paper](http://www.tandfonline.com/doi/pdf/10.1163/156855303769156983?needAccess=true) and [master's thesis](https://pdfs.semanticscholar.org/9037/e7955e4b7c9339dc34bd75e563a86636cca8.pdf).

![Results screenshot](media/results/screenshot.png "Results screenshot")

    /horizoncv                  Python module.
        demo.py                 Running useful real-time demos
        horizon.py              All the important computation
        plotter.py              Plotting graphs
        archive.py              Archived code that might be useful to view in the future.
    /proto
    /media                      Footage for testing
    requirements.txt            Install with $ python install -r 
    runner.py                   Run tests and a demo.


## Usage

Note that you'll need OpenCV compiled with FFMPEG support in order to load videos. See [this script](https://github.com/nsbradford/ExuberantCV/blob/master/installOpenCV.sh) for help. To run a demo:

    $ python runner.py

### Protobuf compilation

After modifying the `horizon.proto` definition, use `protoc` to recompile the python and java files.

    $ cd ./horizon/proto
    $ protoc -I=. --python_out=. horizoncv.proto
    $ protoc -I=. --java_out=.  horizoncv.proto 

## TODO

### Priorities

* Test suite.
* Optimization (find slow part!) to run in real-time (C++/GPU required? )
    * Slowdown when horizon is nearly flat
    * Refinement to run on high-res images
* Search space in [pitch, bank] instead of [slope, intercept] form
    * Why large bank angles isn't found properly? Because the slope-intercept ranges are limited

### Backlog

* Add ZMQ messaging integration.    
* Cleaning and packaging
* Kalman filtering
* Extreme attitude detection (when no horizon is visible)

## Overview

* When the covariance matrix is nearly singular (due to color issues), the determinant will also be driven to zero. Thus, we introduce additional terms to supplement the score when this case occurs (the determinant dominates it in the normal case): where g=GROUND and s=SKY (covariance matrices) 

    F = [det(G) + det(S) + (eigG1 + eigG1 + eigG1)^2 + (eigS1 + eigS1 + eigS1)^2]^-1

* Use algorithm from "Vision-guided flight stability and control for micro air vehicles" (Ettinger et al. ). Intuition: horizon will be a line dividing image into two segments with low variance, which can be modeled as minimizing the product of the three eigenvalues of the covariance matrix (the determinant). In its current form, can run 10000 iterations in 9.1 seconds, or about 30 per iteration at 30Hz. Java performance benefit of 10x would mean 300 per iteration, and moving to 10Hz would leave ~1000 per iteration. The initial optimized search grid is on a 12x12 grid (144 values), which is then refined on a full-resolution image using a gradient-descent-like sampling technique. (requires 4 checks at each step and ~7 steps = ~28, but will but at higher resolution). Total requirements: must be able to run at least ~200 checks/second
* The pitch angle cannot be exactly calculated from an arbitrary horizon line, however the pitch angle will be closely proportional to the percentage of the image above or below the line.
