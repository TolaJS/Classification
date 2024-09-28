"""
author:         Tola Shobande
name:           args.py
date:           4/09/2024
description:    Model hyperparameters and directories
"""

dropout = 0.1                                           # Dropout Rate
batch_size = 64                                         # Batch Size
seed = 7                                                # Chosen seed for reproducibility
lr = 0.01                                               # Learning Rate
gamma = 0.95                                            # Gamma rate
epochs = 10                                             # Number of Epochs
num_classes = 11                                        # Number of Classes
weight_decay = 0.0005                                   # Weight Decay
eval_mode = True                                        # Boolean: True=Evaluating, False=Testing
save_path = "./checkpoint/"                             # Path to save model checkpoints
model_path = "./checkpoint/Model-lr_0.01_bs_64-E10.pth" # Change this to saved model file path
