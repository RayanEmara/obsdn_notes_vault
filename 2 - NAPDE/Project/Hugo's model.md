The main model is `DiffusionAttnUnet1DCond`:

| block_name        | short_desc                                | long_desc                                                                                                            |
|-------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| ResidualBlock     | Basic residual block                      | Combines a main sequential module with an optional skip connection to allow gradients to flow directly.              |
| ResConvBlock      | Residual convolutional block              | A residual block with two convolutional layers, normalization, and GELU activation, including an optional skip layer.|
| SelfAttention1d   | 1D self-attention mechanism               | Applies self-attention to 1D input, enhancing the representation by attending to different positions in the sequence.|
| SkipBlock         | Skip connection block                     | Sequential module with a skip connection that concatenates input and output along the channel dimension.             |
| FourierFeatures   | Fourier feature embeddings                | Projects input into Fourier space to generate high-frequency features using cosine and sine transformations.         |
| Downsample1d      | 1D downsampling layer                     | Reduces the temporal dimension of the input using specified convolution kernel and padding mode.                     |
| Upsample1d        | 1D upsampling layer                       | Increases the temporal dimension of the input using transpose convolution with a specified kernel and padding mode.   |
| nn.Conv1d         | 1D convolutional layer                    | Standard 1D convolutional layer from PyTorch, applies a 1D convolution over an input signal.                         |
| nn.GroupNorm      | Group normalization layer                 | Normalizes across the groups in the input tensor to stabilize learning.                                              |
| nn.GELU           | GELU activation function                  | Gaussian Error Linear Unit activation function, used to introduce non-linearity.                                     |
| nn.Identity       | Identity layer                            | A placeholder layer that returns the input as is, used when no operation is needed.                                  |
| nn.Sequential     | Sequential container                      | A sequential container from PyTorch to stack multiple layers.                                                       |
| nn.Dropout        | Dropout layer                             | Randomly zeroes some elements of the input tensor with probability `p` during training.                              |


