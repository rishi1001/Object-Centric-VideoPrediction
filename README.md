# Combinatorially Generalizing Video Prediction by learning Object- Centric World Models

This repository contains code for processing the Clevrer video dataset and training a Temporal Graph Neural Network (GNN) for bounding box predictions followed by a decoder model(https://arxiv.org/pdf/1804.01622.pdf).

## Prerequisites

- Python 3.6 or higher
- PyTorch
- NumPy
- OpenCV
- PyTorch Geometric
- PyTorch Geometric Temporal

## Steps to Run the Code

1. **Download the Clevrer dataset**
   
   - Download the Clevrer dataset from the official website: [Clevrer Dataset](https://mitibmwatsonailab.mit.edu/research/blog/clevrer-the-first-video-dataset-for-neuro-symbolic-reasoning/).
   - Extract the downloaded dataset to a directory of your choice.

2. **Generate Frames**
   
   - Use the provided script `generate_frames.py` to convert the video dataset into frames.
   - Run the following command:
     ```
     python generate_frames.py
     ```
   - This will generate individual frames from the videos and save them in a separate directory.

3. **Generate Features and Data Points**
   
   - Run the `generate_features.py` script to generate features and data points for training, validation, and testing.
   - Make sure to provide the change the input file locations such as dataset paths, output directories, etc.
   - Example:
     ```
     python generate_features.py
     ```

4. **Run the Encoder Model (Temporal GNN)**
   
   - Navigate to the `Temporal_GNN` folder.
   - Run the `main.py` file to obtain bounding box predictions for the next frames.
   - Make sure to provide the change the input file locations such as dataset paths, output directories, etc.
   - Example:
     ```
     cd Temporal_GNN/
     python main.py
     ```

5. **Train the Decoder Model**
   
   - Run the `train_decoder.py` file to train the decoder model and generate visual frames using the object features.
   - Change the necessary parameters such as training data, model hyperparameters, etc.
   - Example:
     ```
     python train_decoder.py
     ```


Please refer to the code for implementation details and `report.pdf` for architecture and modelling details.

