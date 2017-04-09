# HorizonCV

#### Nicholas Bradford

Real-time horizon detection with computer vision in Python, for UAV applications. Algorithm source found in [paper](http://www.tandfonline.com/doi/pdf/10.1163/156855303769156983?needAccess=true) and [master's thesis](https://pdfs.semanticscholar.org/9037/e7955e4b7c9339dc34bd75e563a86636cca8.pdf)

	/horizoncv
		horizon.py
		plotter.py
	/proto
	/media						Footage for testing
	requirements.txt            Install with $ python install -r 
    runner.py                   Run tests and a demo.


## Usage

Note that you'll need OpenCV compiled with FFMPEG support in order to load videos.

    $ python runner.py

### Protobuf compilation

After modifying the `horizon.proto` definition, use `protoc` to recompile the python and java files.

    $ cd ./horizon/proto
    $ protoc -I=. --python_out=. horizon.proto
    $ protoc -I=. --java_out=.  horizon.proto 

## TODO

### Priorities

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

