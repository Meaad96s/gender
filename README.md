# Gender-Classification

## Download packages
```
pip3 install glob3
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pillow
pip install matplotlib
pip install scikit-image
git clone https://github.com/1adrianb/face-alignment
pip install -r requirements.txt
python setup.py install
pip install sklearn

On Conda Prompt
conda install -c anaconda mkl
conda install -c pytorch pytorch torchvision
```
# Steps
### Preprocessing
## Read images

RGB (Red, Green, Blue) are 8 bit each.
The range for each individual colour is 0-255 (as 2^8 = 256 possibilities).
The combination range is 256*256*256.

By dividing by 255, the 0-255 range can be described with a 0.0-1.0 range where 0.0 means 0 (0x00) and 1.0 means 255 (0xFF).


## Convert to grey
Convert to grey scale

## Resizing
Resize pictures

## Spatial Scale
perform standard scaler
## Haar

## Annotation 
Label pictures
### Feature Extraction 
## HOG

## LBP

### Dimention Reduction 
## PCA

## LDA
### Classifier 

## SVM

## KNN

## Normalization
### Experiments

| Experiment No.        | Preprocessing          | Feature Extraction  |Dimention Reduction | Classifier | Evaluation Metric |
| ----------------------|:----------------------:| -------------------:|-------------------:|-----------:|------------------:|
| 1                     | Convert To Grey, Haar  | HOG                 | PCA                | SVM        | 
| 2                     | Convert To Grey        | HOG                 | LDA                | SVM        |
| 3                     | Convert To Grey        | LBP                 | PCA/LDA            | SVM        |
| 4                     |-                       | -                   | -                  | CNN        | 

