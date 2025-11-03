# Steganography Discriminator using CNN & RL

1. CNN_Disc : ResNet detector for reward return during Reinforcement Learning training.
2. filter_ex : Code for experimental application of the Laplacian filter.
3. PPOTrainer : PPO trainer for cumulative training of the Reinforcement Learning model.
4. ResNet_nofilter : Model for image classification testing without an adaptive filter applied.
5. ResNet_yesfilter : Model for measuring the misclassification rate.
6. RL_proto : (AC + PPO) + CNN Model, where AC (Actor-Critic) selects the attention mask position, PPO (Proximal Policy Optimization) handles cumulative training, and CNN (Convolutional Neural Network) performs steganography detection.

# Model Architecture
<img width="1814" height="825" alt="image" src="https://github.com/user-attachments/assets/1ddcad00-0b8f-479c-b9d1-3baf0b5bdfb0" />

# Experiment
<img width="1888" height="1004" alt="image" src="https://github.com/user-attachments/assets/141a1714-c125-464c-ba23-6815002b16d2" />

# Result
| Metric | ResNet-18 with Adaptive Filter | ResNet-18 with G4L Filter |
| :--- | :---: | :---: |
| **Accuracy** | 87.60% | 85.72% |
| **Precision** | 0.83 | 0.84 |
| **Recall** | 0.94 | 0.88 |
| **F1 Score** | 0.88 | 0.86 |
