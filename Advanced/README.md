# Proposed Model

The proposed approach includes one primary novelty for MRI preprocessing using the Intermediate State Generator (IS-Gen). 
The IS-Gen is a new way of normalizing slice gaps across varying MRI scans. There are existing co-registration tools, but these use linear interpolation to normalize slice gaps.
The IS-Gen uses an intelligent adversarial algorithm to produce high quality MRI scans. We are working on conclusively proving that the IS-Gen is able to produce
more informative MRI scans compared to the linear interpolation. 

The training MRI scans were preprocessed an converted to _.npy_ files to allow for faster training. This preprocessing uses the IS-Gen model and the full pipeline
is included in ```test-enhanced-resnet10.ipynb```. 

Finally, the produced _.npy_ files were used to train the model. The training script is provided in ```enhanced-resnet10-only-tumor.ipynb```. 
