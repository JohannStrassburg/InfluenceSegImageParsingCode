========================================================================
SuperParsing Code
Created by Joe Tighe (jtighe@cs.unc.edu) 9/3/2010

This MATLAB code implements parsing system described in the following paper:

Joseph Tighe and Svetlana Lazebnik, "SuperParsing: Scalable Nonparametric 
Image Parsing with Superpixels," European Conference on Computer Vision, 2010.
========================================================================

Version 1.0 (Updated 5/10/2012)

This is a beta version of the system. The system needs a number of libraries to work and we are in the process of negotiating with the library owners to inclued the libraries here. As of now you'll need to download the libraries from the following locations and compile them:

I sugest copying them into the im_parser\Libraries folder. If you don't you'll need to open SetupEnv.m and modify it to point the the locations you downloaded the files.

MRF Graph Cut Code (place in im_parser\Libraries\gco-v3.0):
code: http://vision.csd.uwo.ca/code/gco-v3.0.zip
website: http://vision.csd.uwo.ca/code/

LabelME Toolkit (place in im_parser\Libraries\LabelME):
code: http://labelme.csail.mit.edu/LabelMeToolbox/LabelMeToolbox.zip
website: http://labelme.csail.mit.edu/

If you've set up you directory structure properly you just need to go to the base directory and run:

RunFullSystem

This should compile and setup the enviroment then generate the features, train the classifiers if needed and parse the dataset. It will also evaluate the parse and generate websites of the parser output.


Thanks to the following libray authors for alowing me in include their code:

Gist (place in im_parser\Libraries\gist):
code: http://people.csail.mit.edu/torralba/code/spatialenvelope/gist.zip
website: http://people.csail.mit.edu/torralba/code/spatialenvelope/

Boosted decesion tree code. 
If you'd like to download it for other uses it can be found here:
http://www.cs.uiuc.edu/homes/dhoiem/

Segmentation Code is now included. If you want to download it yourself go here:
code: http://people.cs.uchicago.edu/~pff/segment/segment.zip
website: http://people.cs.uchicago.edu/~pff/segment/
unzip into: superparser/Libraries/segment but don't over-write any of the modified cpp files currently there

Texton Filter Code (place in im_parser\Libraries\anigaussm):
code: http://staff.science.uva.nl/~mark/downloads/anigaussm.zip
website: http://staff.science.uva.nl/~mark/downloads.html#anigauss
website: http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html