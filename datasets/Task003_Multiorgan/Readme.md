# Multiorgan segmentation

# Classes:
- 0 -> Background
- 1 -> Spleen
- 2 -> Pancreas
- 3 -> Left kidney
- 4 -> Gallbladder
- 5 -> Ssophagus
- 6 -> Liver
- 7 -> Stomach
- 8 -> Duodenum

# Folder Structure:
## CT --> CT scan of different patients
## GroundTruth --> Manual Segmentation that is made by an expert
## Predictions --> Algorithms Prediction
Each Approach will have a folder and the predicted segments are the same as the CT scan name. For example, prediction of AlgA is Predictions/AlgA/1.nii.gz


# Licence: "CC-BY"