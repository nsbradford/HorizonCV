# HorizonCV

#### Nicholas Bradford

Real-time horizon detection with computer vision in Python, for UAV applications. Algorithm source found in [paper](http://www.tandfonline.com/doi/pdf/10.1163/156855303769156983?needAccess=true) and [master's thesis](https://pdfs.semanticscholar.org/9037/e7955e4b7c9339dc34bd75e563a86636cca8.pdf)

### Requirements

* Python 3
* OpenCV
* NumPy

### Not yet implemented

* Search space in [pitch, bank] instead of [slope, intercept] form
* Debug:
	* Slowdown when horizon is nearly flat
	* Why large bank angles isn't found properly? Because the slope-intercept ranges are limited
* Cleaning and packaging
* Kalman filtering
* Extreme attitude detection (when no horizon is visible)

