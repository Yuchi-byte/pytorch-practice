## Training neural networks to classify the fashion-MNIST dataset. 

0. My first attempt was a very simple fully connected linear-ReLU stack with SGD optimizer (achieved 80% accuracy in 10 epochs). Thereafter, a few changes were made to increase the accuracy significantly: 
1. Using Adam optimiser that is commonly considered in the field to be one of the best optimisers. The batch size is also decreased from 64 to 32 to encourage the gradient descent algorithm to pick up the important nuances that would otherwise be averaged out in a larger batch. 
2. Adding convolution layers (together with ReLU activation and max pooling) before the flattening and being passed to the FC. This means using CNN. 
3. Applying dropout to regularize and generalise. After the first two steps, significant overfitting was observed from a U-shaped test loss despite the test accuracy kept increasing (This is because even though the majority class is still predicted correctly, the probabilities assigned to the other classes by the nn increases which contributes to higher cross entropy loss). Therefore, dropout is applied. 
4. After the first three steps, the nn reaches an accuracy just shy of 90% after 20 epochs. Therefore, multiple batch normalisation layers were added. The resulting accuracy is 92% after 20 epochs.
5. Literature review suggests that using vgg-like architectures tend to achieve high accuracy. So the vgg-like architecture is implemented. The final accuracy is 93% after 30 epochs. The vgg-like architecture contains multiple blocks of convolution layers with small kernels (3x3) followed by fully connected layers for classification.


A note on the number of parameters to be learned by the nn: 
* A convolution layer with filter size F*F, stride = 1, zero padding and output channel (the same as the number of filters) C_out has a number of parameters = F * F * C_in * C_out + C_out. The tensor's size changed from (B, C_in, H, W) at input, eg. (32, 1, 28, 28), to (B, C_out, 28-F+1, 28-F+1).
* Batch normalisation has 2*C_out number of parameters. I think of batch normalisation to be applied on each channel, and at every channel, the mean and variance are calculated. This results in 2 * C parameters.
* max pooling doesn't introduce any parameters.
* FC. number of parameters = (input size + 1) * output size. For the first linear layers that follows the convolution stack, the input size = C_out * H * W (both H and W would have already been subtracted F-1 multiple times). The output size of the linear layer is to be determined by the human. 
