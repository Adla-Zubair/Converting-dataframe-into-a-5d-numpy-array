# Converting dataframe into a 5d numpy array for LIMO
 The code is to convert the dataframe into a numpy array for LIMO

LIMO requires data to be presented in a specific way, that is, in 
Brain Imaging Data Structure (BIDS) standard. The input is preprocessed and segmented data that 
is stored as .mat file or as .set file

LIMO generate outputs generated at levels :
 for example - (n_channels x n_timeframes x n_variables) matrix

The dataframe used here in the code has columns : channels(21), features (20 columns), 
subject name(25), subjectclassification(2), sleep stage.

The code converts the dataframe into a 5d array with the first axis focusing on the 
subject classification, followed by 5 sleep stages, followed by the channels, each subject
and then the features.
This gives a 5d array of dimension (2,5,21,21,20).

References : https://www.fieldtriptoolbox.org/getting_started/limo/

