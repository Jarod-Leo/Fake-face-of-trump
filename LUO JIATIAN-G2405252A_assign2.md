
# Developing a GAN for Generating Fake Images of Donald Trump
## 1. Dataset
For this project, I downloaded a dataset containing 3020 color images of Donald Trump from [Kaggle Dataset](https://www.kaggle.com/datasets/mbkinaci/trump-photos), with each image resized to 64x64 pixels. The Creator of this dataset collected Trump's face images from main social media and website, ensuring varied facial expressions, lighting conditions, and angles to enable the network to generalize well across different representations, which makes it a effective training dataset.Because of the images miniaturization and diversity, I choose it as my dataset for reducing computation and improving model robustness.

## 2. Network Architecture
To generate realistic images of Donald Trump, I selected a Generative Adversarial Network (GAN) framework, as it has proven effective for image synthesis tasks. Specifically, I employed the Deep Convolutional GAN (DCGAN) architecture, which leverages convolutional layers to produce high-quality images and is relatively lightweight, suitable for my dataset size.

- **Generator Network**
The generator transforms a random vector(sampled from a standard normal distribution) into a 64x64 image. To achieve this, the generator consists of 5 network blocks. In the first four blocks, each convolutional layer is followed by batch normalization and a ReLU activation function. The final block consists of a convolutional layer followed by a tanh activation function to produce output values between -1 and 1, aligning with the normalized image data range.The process of handling the noise vector in the first four convolutional blocks is essentially an upsampling process, which gradually increases the image area and reduces the image feature maps. Therefore, the parameters for each layer are set as follows:

<!-- 让表格居中显示的风格 -->
<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="times" size=2.>Table1 Parameters of each Convolutioanal layer in the Generator</font></p>

<div class="center">

|   Convolutional layer   |   Kernal Size  |        Stripe         |     Padding  |
|  :---:                  |  :---:         |  :---------------:    |   :-------:  |
|    1                    |    4           | 1                     |     0         |
|    2                    |    4           |      2                 |     1         |
|    3                    |    4           |        2               |          1    |
|    4                    |    4           |        2               |        1      |
|    5                    |    4           |        2               |        1      |
</div>
the first block take the latent vector and transformed it from 1x1 spatial dimension to 4x4 spatial dimension and reduce the feature maps channel at the same time. The second block produce a 8x8 spatial dimension, through repeated iterations of this process, we ultimately achieve spatial dimensions of 64x64 with 3 channels in the resulting images.

- **Discriminator**
The discriminator also contains 5 convolutional layers, the first 4 convolutional layers each followed by batch normalization and Leaky ReLU activation, and the last convolutional layer followed by a sigmoid layer, as we need the last layer outputs a single probability value indicating the likelihood that an input image is real. And the entire process of discrimination is essentially a procedure of downsampling the input images. So I set up the parameters of convolutional layer below to achieve the goal:
<!-- 让表格居中显示的风格 -->
<style>
.center 
{
  width: auto;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<p align="center"><font face="times" size=2.>Table2 Parameters of each Convolutioanal layer in the Discriminator</font></p>

<div class="center">

|   Convolutional layer   |   Kernal Size  |        Stripe         |     Padding  |
|  :---:                  |  :---:         |  :---------------:    |   :-------:  |
|    1                    |    4           | 2                     |     1         |
|    2                    |    4           |      2                 |     1         |
|    3                    |    4           |        2               |          1    |
|    4                    |    4           |        2               |        1      |
|    5                    |    4           |        1               |        0      |
</div>

## Additional Resource
To support the training process and improve model performance, I require the following resources:
- **Hardware:** Access to a GPU(Preferably an NVIDIA GPU with CUDA Operators) to speed up the training process. Because the task of training a DCGAN model is computational and need to experiment significant numerous parameter sets to achieve favorable consults, using a GPU can greatly  can expedite the convergence of results.
- **Software:** I need Pytorch framework for implementing and training the DCGAN, besides I need Numpy and Pandas libraries for handling data structures.

## Experiment
Based on the neural network architecture described above, I have reproduced the entire code and conducted training on my own computer.In the experiment, I trained the DCGAN model on the dataset of Donald Trump images for 1000 epochs, using a batch size of 64.Ultimately, I obtained a reasonably satisfactory generated image of Donald Trump.And Further details can be found in my GitHub repository.
After 1000 epochs of training, the generated images are shown below, These images, for the most part, are relatively easy to identify as Donald Trump.
<div style="text-align: center;">

![images](generate_trump\sample_1000_v2.png)
</div>

And the generator loss curves and discriminator loss curves are depicted in the figure below.
<div style="text-align: center;">

![loss_curves](loss\loss2.png)
</div>
we can see after 1000 epochs, the losses of the generator and discriminator converge around 0.68.

## Design and Thoughts
The choice of DCGAN is motivated by its proven success in generating high-quality images. The use of convolutional layers in both the generator and discriminator allows the network to learn spatial hierarchies of features, which is crucial for generating realistic images. My design approach emphasizes a balance between model complexity and computational feasibility. Here are the key considerations behind each component:
- **Generator**
I choose to utilize 5 convolutional layers because too deep a network could lead to overfitting, while a shallow network might fail to capture the intricate details of Trump’s facial features.Besides batch normalization layer can stabilize the generator output and prevent common GAN issues such as mode collapse.
- **Discriminator**
The discriminator architecture includes equal depth to the generator with batch normalization to efficiently distinguish real from synthetic images. The distinction between the two lies in the fact that the discriminator uses Leaky ReLU as its activation function as Leaky ReLU activations in the discriminator help handle the sparse gradients issue, common in GAN training. 
- **Training Strategy**
I use the Adam optimizer with a learning rate of 0.0002 for both the generator and 0.00002 for discriminator, which has been found to work well in GAN training.Because the discriminator is strong, if we set the learning rate same as generator it would easily detect the generator's outputs, leading to the generator failing to improve. By setting the discriminator's learning rate smaller, the generator has more opportunities to improve before the discriminator becomes too accurate, fostering more stable and balanced training.