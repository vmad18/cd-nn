# CD-NN
- Early detection of 8 cancer types from human blood plasma samples via a neural network. 
- Inspiration from research paper published in Science, "Detection and localization of surgically resectable cancers with a multi-analyte blood test."
- Used in local science fair, where I was interviewed for ISEF. 
- Code presented is a very "cleaned up" version of the colab notebook used for the project.  

# Cancer Detect Neural Network
- 4 Layer Neural Network
- Input: 39 dimensional vector representing protein levels of blood plasma proteins
- 2 Hidden Layers were used to ensure a function that will generalize the data
- Output: 2 dimensional vector representing the probabilistic likelihood of the presence of cancer using softmax
- Model was hyperparameter tuned and evaluated using Nested Cross Validation (CV)
- Nested CV also helps to mitigate overfitting

# Results 
- CD-NN's sensitivities for stages I, II, and III cancers were **94%**, **92%**, and **92%**. 
- Improvement over the past paper's **43%**, **75%**, and **78%** sensitivities for stages I, II, and III cancers. 

# Requirments
- Tensorflow, latest stable
- Sci-Kit Learn, latest stable
- Numpy, latest stable
- Pandas, latest stable

# Paper & Dataset
- Researchers used logistic regression for detection of cancer.
- https://www.science.org/doi/10.1126/science.aar3247
