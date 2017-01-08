===================================================================================
Copyright Information
===================================================================================

InfluenceSegImageParsing Code
Created by Johann Straßburg (johann.strassburg@udo.edu) 2014

This code implements the test code used for the following paper:

"On the Influence of Superpixel Methods for Image Parsing"
Johann Strassburg, Rene Grzeszick, Leonard Rothacker, Gernot A. Fink
Proc. International Conference on Computer Vision Theory and Applications (Visapp), 2015.

It was originally used in the master thesis:
"Segmentierungsverfahren für Bildparsing in natürlichen Szenen", Johann Strassburg (2014)
http://patrec.cs.tu-dortmund.de/cms/en/home/Publications/Theses/index.html  

===================================================================================
This code is based on previous work and code by (and not limited to):

Folder: "SuperParsing":
Joseph Tighe and Svetlana Lazebnik, "SuperParsing: Scalable Nonparametric 
Image Parsing with Superpixels," European Conference on Computer Vision, 2010.

Folder: "Edge_Avoiding_Wavelets":
Fattal,  R.  (2009). "Edge-avoiding  wavelets  and  their  ap-
plications." ACM  Transactions  on  Graphics  (TOG)

Folder: "Saliency":
B. Schauerte, R. Stiefelhagen, "How the Distribution of Salient 
Objects in Images Influences Salient Object Detection". In 
Proceedings of the 20th International Conference on Image Processing
(ICIP), 2013.

For more information relate to code-documentation within the program-structure 
and/or notes within the thesis and/or the paper(s)

===================================================================================

Instructions to perform SuperParsing with different superpixels methods

===================================================================================

Contents:
** 1. Experiment File Structure
** 2. SuperParsing
** 3. Superpixel Creation
** 4. Further Tools

Folder Overview:
*Edge_Avoiding_Wavelets
*Saliency
*segparsing
*SuperParsing

========================================
	1. Experiment File Structure
========================================

This section gives an overview how an Experiment can be structured for SuperParsing evaluation as follows:
****************************************************
+parent folder+
	+subfolder 1+
	+subfolder 2+
		+subsubfolder+ #folder description
	<file>
	*folder/file provided by database* #(other folders/files are/needs to be created by e.g. superParsing algorithm (see section 2)) 
****************************************************


+Database+ 	#for example 'Barcelona' for 'Barcelona' dataset
	+Images+	#folder containing Images of the dataset
	+Experiments+	#folder containing different superParsing experiments (evaluation, superpixel configurations etc.)
		+experiment_1+ #one experiment e.g. 'SLIC_50_1_1' for an experiment using SLIC superpixels
		+experiment_2+
			+*GeoLabels*+		#folder containing geometric labels to images
				+SP_Desc_k200+		#folder containing geometric labels to superpixels 
			+*SemanticLabels*+	#folder containing semantic labels to images
				+SP_Desc_k200+		#folder containing semantic labels to superpixels
			<*TestSet1.txt*> #file linking to the test set of images (Attention: check correct addressing for Linux/Windows computers)
			+Python+	#folder for Python output created by superParsing.py in segparsing project
			+Data+ #contains most of produced output by SuperParsing
				+Base+
					+GeoLabels+		#contains processed geometric label information of testset
					+SemanticLabels+	#contains processed semantic label information of testset
					+MRF+			#contains raw SuperParsing results for testset
					+RetrievalSet		#contains retrieval sets for each image of testset
					<ResultsMRF>		#Summarized results of SuperParsing
				+Descriptors+	
					+Global+	#contains global descriptors of all Images
							#can be created once for one database and copied to new experiments
					
					+SP_Desc_k200+	#contains all superpixel descriptors as well as superpixels
							#name originally from from Graph-Based configuration within original SuperParsing configuration
							#remains the same if parameters are not changed (also if other superpixels are used, see section 2)
						+super_pixels	#folder contains superpixels
							



========================================
	2. SuperParsing
========================================
To perform SuperParsing go to SuperParsing/im_parser:

++ 1.Execution ++
Files 'RunSift*.m', 'RunBarca*.m' can be run to perform SuperParsing on the SiftFlow/Barcelona Dataset.
Input argument is the experiment folder name e.g. 'Quick_Shift_10_48_0.05'

++ 2.Options ++
Options can be found in SuperParsing/im_parser/DataSpecific:
Files 'RunSiftFlow*.m', 'RunBarcelona*.m', called by executed file in 2.1 inherits options as well as Data paths

++ 3. Example execution ++
- 1 -
**** Perform SuperParsing on a Quick Shift experiment with precalculated ****
**** superpixels on the Barcelona Dataset                                **** 
nohup nice /vol/local/amd64/matlab2013b/bin/matlab -nodisplay -nodesktop -r "RunBarca('Quick_Shift_10_48_0.05'); exit;"

- 2 -
**** Perform SuperParsing on an experiment with Ground Truth superpixels  ****
**** superpixels on the Barcelona Dataset                                ****
nohup nice /vol/local/amd64/matlab2013b/bin/matlab -nodisplay -nodesktop -r "RunGTBarca('GroundTruth'); exit;" 

- 3 -
**** Perform SuperParsing using and creating superpixels by the          ****
**** Graph-Based approach on the SiftFlow dataset                        ****
nohup nice /vol/local/amd64/matlab2013b/bin/matlab -nodisplay -nodesktop -r "RunSift('Graph_based'); exit;"


========================================
	3. Superpixel Creation
========================================

++ 1. Groundtruth superpixels ++

Use SuperParsing algorithm (section 2) to create Groundtruth segments and perform SuperParsing.
See section 2.3.2 for an example execution on the Barcelona dataset

++ 2. Efficient Graph-Based image segmentation ++

Use SuperParsing algorithm (section 2) to create Graph-Based segments and perform SuperParsing.
See section 2.3.3 for an example execution on the SiftFlow dataset

++ 3. SLIC ++

Go to segparsing folder and execute superParsing.py in 'main'-folder (you may need to copy file into parent folder, if using it not as a project import (e.g. in eclipse))
See usage of superParsing script for parameters input.
Change global path variables in segparsing/utils/utils.py.

++ 4. Quick Shift ++

Go to segparsing folder and execute superParsing.py in 'main'-folder (you may need to copy file into parent folder, if using it not as a project import (e.g. in eclipse))
See usage of superParsing script for parameters input.
Change global path variables in segparsing/utils/utils.py.

++ 5. Grid based segmentation ++

Go to segparsing folder and execute superParsing.py in 'main'-folder (you may need to copy file into parent folder, if using it not as a project import (e.g. in eclipse))
See usage of superParsing script for parameters input.
Change global path variables in segparsing/utils/utils.py.

++ 6. Saliency ++

Go to Saliency/region_contrast_saliency-master folder.
Copy images to process into a subfolder 'Images' (or change path in segment_images.m).
Create salience superpixels by:

1. Use segment_images.m to create saliency matrices in folder 'segments' from images in folder 'Images'.

2. Use heatsal.py to convert saliency matrices from folder 'segments' into heatmap-images in folder output.

3. Use images from output folder to create superpixel (e.g. Graph-Based segmentation through implementation within SuperParsing (see section 3.2.).

4. Use created superpixels in SuperParsing code (Attention: if created by SuperParsing code, do not forget to change input images back to normal (not heatmap images) to create segment-features etc.).



++ 7. Edge Avoiding Wavelets ++

- 1. Creating Edge Avoiding Wavelets superpixels based on argmax of scaling functions -

Go to Edge_Avoiding_Wavelets/eaw_code folder.
Copy images to process into a subfolder 'Images' (or change path in eaw_superpixels.m).
Use eaw_superpixels script to create superpixels for images (see eaw_superpixels.m for output information).
ATTENTION: selection of many scales and/or input of big images and/or many images can results in a huge amount of files.
TO SHRINK AMOUNT OF FILES: deactivate creation of images and/or shrink scaling functions to one file per scale (see section 4.1.1).

- 2. Relabel EAW-superpixel to merge smaller regions to next bigger regions in order to have only fully connected areas -

Go to Edge_Avoiding_Wavelets/eaw_relabeling folder.
Use relabel.py to relabel created EAW superpixels.


========================================
	4. Further Tools
========================================

++ 1. Weighting EAW superpixels results by scale functions ++

The results given by EAW superpixels can be improved by weighting superpixels with scaling function weights.
The following steps are needed:

- 1 -
Go to Edge_Avoiding_Wavelets/concat_eaw.
Use eaw_concat.py to concatenate created scaling functions to one file according to superpixel indices (see ReadMe for details).

- 2 -
Go to Edge_Avoiding_Wavelets/eval_eaw_*database*.
Use e.g. eaw_ev.py to evaluate results by weighting superpixel (label selection).
For further details see README

++ 2. Utils ++

Go to segparsing/utils to get access to some tools.

- 1. Statistics -

Use Statistics.py to analyze superpixels (e.g. size etc.) of multiple experiments.

- 2. Results -

Use evalcopy.py to copy Results from multiple Experiments to one place.

- 3. Utils -

Use utils.py for different processing tools (e.g. reading of mat-files).





