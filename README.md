# Steganography Discriminator using CNN & RL

1. CNN_Disc : ResNet detector for reward return during Reinforcement Learning training.
2. filter_ex : Code for experimental application of the Laplacian filter.
3. PPOTrainer : PPO trainer for cumulative training of the Reinforcement Learning model.
4. ResNet_nofilter : Model for image classification testing without an adaptive filter applied.
5. ResNet_yesfilter : Model for measuring the misclassification rate.
6. RL_proto : (AC + PPO) + CNN Model, where AC (Actor-Critic) selects the attention mask position, PPO (Proximal Policy Optimization) handles cumulative training, and CNN (Convolutional Neural Network) performs steganography detection.
