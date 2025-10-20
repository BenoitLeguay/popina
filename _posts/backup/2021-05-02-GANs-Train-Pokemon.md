---
layout: post
title: GANs Training Journey Pokemon
github: https://github.com/BenoitLeguay/GAN_IconClass
---

| ![results.png]({{site.baseurl}}/images/gans/results.png) |
| :------------------------------------------------------: |
|                     *Result example*                     |

<br />

GANs are a framework for teaching a DL model to capture the training data’s distribution so we can generate new data from that same distribution. GANs were invented by Ian Goodfellow in 2014 and first described in the paper [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). They are made of two distinct models, a *generator* and a *discriminator*. The job of the generator is to spawn ‘fake’ images that look like the training images. The job of the discriminator is to look at an image and output whether or not it is a real training image or a fake image from the generator. During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the real and fake images. The equilibrium of this game is when the generator is generating perfect fakes that look as if they came directly from the training data, and the discriminator is left to always guess at 50% confidence that the generator output is real or fake. (@pytorch)

<br />

![gan-schema.svg]({{site.baseurl}}/images/gans/gan-schema.svg)

<br />

The goal of this training journey is not to create the next Pokemon generation but to observe and compare GANs performance through modification in its architecture, training components, hyper parameters etc.. 

<br />

We will compare 3 GANs versions: 

- Deep Convolutional GANs
- Wasserstein GANs with gradient penalty
- Auxiliary Classifier GANs

<br /><br />

## Dataset

The Pokemon sprites dataset has advantages, uniform low resolution images with a wide diversity. 

| ![gan-dataset.png]({{site.baseurl}}/images/gans/gan-dataset.png) |
| :----------------------------------------------------------: |
|                       *dataset batch*                        |

The images are 64x64 resolution, with 3 channels. The only preprocessing made is a minmax scale. The goal of this operation is to normalize images between -1 and 1, this can make training faster and reduces the chance of getting stuck in local optima.

$$x_{norm} = 2*(\frac{x - x_{min}}{x_{max} - x_{min}}) - 1$$

<br /><br />

## A) DCGANs

DCGANs is simple version of GANs (described hereinabove) that uses convolutional layers for both the discriminator and the generator. Let's see how our DCGAN works on this task.   <br /><br />

The generator tries to minimize the following function while the discriminator tries to maximize it:

$$E_x[log(D(x))] + E_z[log(1 - D(G(z)))]$$

<br />

#### **1) Discriminator:**

| ![DCGAN discriminator.png]({{site.baseurl}}/images/gans/DCGAN discriminator.png) |
| :----------------------------------------------------------: |
|                 *Discriminator architecture*                 |

<br />

Each convolutional block consists of a Conv2d/ Batch Normalization/ LeakyReLu sequence. The output of the *Discriminator* is then fed into a sigmoid function. This gives us a score between 0 and 1, that is, the likelihood to be either a real or a fake sample.   <br />

$$L_D = E_x[log(D(x))] + E_z[log(1 - D(G(z)))]$$

It can be seen as a sum of 2 binary cross entropy loss where labels are ones for the real examples and zeros for the fakes.  

<br />



#### **2) Generator:**

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png) |
| :----------------------------------------------------------: |
|                   *Generator architecture*                   |

<br />

This network has a Batch Normalization, Dropout, LeakyReLu layers at each convolutional block. We'll discuss the different upsample method in another part. Since my real images are normalized between -1 and 1, I use the Tanh function as output activation. The network takes as input a random vector of size $$Z_{dim} = 100$$ sampled from the normal distribution. The generator function maps the latent vector Z to the data space. <br />

The loss function is different from the minimax one we defined before, but the general idea remains. This comes from the original paper, it avoids vanishing gradient early in training.  The $$D(x)$$ terms is removed since it is invariant to the generator.  

 $$L_G = E_z[-log(D(G(z)))]$$

<br />



#### **3) Unit test**

As a unit test, I like to make my GAN to reproduce a single image. This is also a good comparison tool across multiple GANs architectures, when talking about learning pace mostly. <br />

| ![dcgan-1p-real.png]({{site.baseurl}}/images/gans/dcgan-1p-real.png) |
| :----------------------------------------------------------: |
|                        *Real dataset*                        |

 <br />

Our GANs is fed with the same image during the whole training. The dataloader contains 1000 times the same image. <br />

| ![dcgan-1p-10e.png]({{site.baseurl}}/images/gans/dcgan-1p-10e.png) | ![dcgan-1p-50e.png]({{site.baseurl}}/images/gans/dcgan-1p-50e.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         *10 epochs*                          |                         *50 epochs*                          |

| ![dcgan-1p-150e.png]({{site.baseurl}}/images/gans/dcgan-1p-150e.png) |
| :----------------------------------------------------------: |
|                         *150 epochs*                         |

<br />

Now we know that our DCGANs flow works we can train it on the whole dataset.

<br /><br />

#### 4) Training example

| ![dcgan-10e.png]({{site.baseurl}}/images/gans/dcgan-ex-e10.png) | ![dcgan-50e.png]({{site.baseurl}}/images/gans/dcgan-ex-e50.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         *10 epochs*                          |                          50 epochs                           |
| ![dcgan-100e.png]({{site.baseurl}}/images/gans/dcgan-ex-e100.png) | ![dcgan-250e.png]({{site.baseurl}}/images/gans/dcgan-ex-e250.png) |
|                          100 epochs                          |                         *250 epochs*                         |
| ![dcgan-400e.png]({{site.baseurl}}/images/gans/dcgan-ex-e400.png) | ![dcgan-800e.png]({{site.baseurl}}/images/gans/dcgan-ex-e800.png) |
|                         *400 epochs*                         |                         *800 epochs*                         |

<br />

| ![dcgan-discri.svg]({{site.baseurl}}/images/gans/dcgan-ex-dloss.png) | ![dcgan-gen.svg]({{site.baseurl}}/images/gans/dcgan-ex-gloss.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|              *Discriminator Loss over updates*               |                *Generator Loss over updates*                 |

<br />

| ![dcgan-ex-facc.png]({{site.baseurl}}/images/gans/dcgan-ex-facc.png) | ![dcgan-ex-racc.png]({{site.baseurl}}/images/gans/dcgan-ex-racc.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|          *Discriminator accuracy on fake examples*           |          *Discriminator accuracy on real examples*           |

| ![dcgan-ex-fid.png]({{site.baseurl}}/images/gans/dcgan-ex-fid.png) |
| :----------------------------------------------------------: |
|           *Frechet Inception Distance over epochs*           |

 <br />

The Frechet Inception Distance score, or FID for short, is a metric that calculates the distance between feature vectors calculated for real and generated images. It uses the *Inception_v3* deep neural network to extract a latent space of every images.

<br /><br />

  

## B)  WGANs with Gradient Penalty

Wasserstein Generative Adversarial Networks are an extension to the Vanilla GAN, the main modification lies in the Discriminator, here named **Critic**, network objective. Instead of estimating a probability of being a real sample, the network outputs a score that shows the realness of the input. It can be compared to the Value function $$V(s)$$ used in Reinforcement Learning. Hence, the loss is modified accordingly, but first let's talk about the Wasserstein distance. 

Indeed, the mathematical idea behind WGANs is the Wasserstein distance (or earth mover distance). That is, a measure of distance between 2 distributions. In this context, we can see the **Critic** as a function approximation, useful for the Wasserstein distance. 

*[...] we define a form of GAN called Wasserstein-GAN that minimizes a reasonable and efficient approximation of the EM distance, and we theoretically show that the corresponding optimization problem is sound.* (https://arxiv.org/abs/1701.07875)

<br />

It is fair to see the WGAN loss function as an approximation of the Wasserstein distance. Indeed, the original paper shows that we can write the Wasserstein formula down as.

$$W_{dist}(P_{real}, P_{fake}) = \frac{1}{K}\sup_{\lVert f \rVert_L<1}E_{x \sim P_{real}}[f(x)] - E_{x \sim P_{fake}}[f(x)]$$

In the context of GANs, the above formula can be rewritten by sampling from z-noise distribution and replacing $$f$$ by the Critic function, $$C_w$$:

$$W_{dist}(P_{real}, P_{fake}) = max_{w\in W}[E_{x \sim P_{real}}[C_w(x)] - E_{z \sim \mathcal{N_{\mu, \sigma}}}[C_w(G(z))]]$$

The **Critic** network tries to approximate $$f$$. A good estimation is the key point of this architecture, this is why you often see people updating severals time the Critic for one Generator update. Yet, this formula comes with a constraint. The gradient penalty answers it back. 

 <br />

So, the intuition between the gradient penalty comes from the calculation of the Wasserstein distance. I won't cover the full maths here, but in a few words, we need our **Critic** function to be 1-Lipschitz (i.e gradient norm at 1 everywhere). To ensure this we add this gradient penalty term in our critic loss function. 

Another WGAN version tries to guarantee this constraint by clipping the weight update. Its stability really depends on the clipping hyper parameter and thus the gradient penalty version is considered more reliable. Though, it is computationally expensive. 

<br /><br />

#### **1) Critic**

| ![DCGAN discriminator.png]({{site.baseurl}}/images/gans/DCGAN discriminator.png) |
| :----------------------------------------------------------: |
|                    *Critic architecture*                     |

As we have seen above, the architecture of the WGANs' Critic does not differ from the Discriminator, except in the output activation, being the Identity function here. It is important to mention that I remove the **Batch Normalization Layer** because it takes us away from our objective.

<br />

*[...] but batch normalization changes the form of the discriminator’s problem from mapping a single input to a single output to mapping from an entire batch of inputs to a batch of outputs. Our penalized training objective is no longer valid in this setting, since we penalize the norm of the critic’s gradient with respect to each input independently, and not the entire batch [...]* (https://papers.nips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf)

Perhaps most importantly, the loss of the **Critic** appears to relate to the quality of images created by the Generator.

Specifically, the lower the loss of the **Critic** when evaluating  generated images, the higher the expected quality of the generated  images. This is important as unlike other GANs that seek stability in terms of finding an equilibrium between two models, the WGANs seeks convergence, lowering generator loss.

<br />

$$L_c=E_z[C(G(z))]-E_x[C(x)]+\lambda(\lVert \nabla_{\hat{x}}D(\hat{x}) \rVert_2-1)²$$

*with $$\hat{x} = \epsilon x+(1-\epsilon) G(z)$$ and $$\lambda$$ being the gradient penalty factor*

<br /><br />

#### **2) Generator:**

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/DCGAN generator.png) |
| :----------------------------------------------------------: |
|                   *Generator architecture*                   |

<br/>

Not much to say here, it does not differ from the **DCGANs** one.  

Concerning the loss, our generator wants to maximize the critic output, therefore we minimize the negative  average of the critic score among fake samples. The $$x$$ associated term disappears since is not directly optimizing with respect to the real data. 

$$L_G= - E_z[C(G(z))]$$

<br /><br />

#### **3) Unit test**

**Real samples**

| ![wgan-ex-1p-real.png]({{site.baseurl}}/images/gans/wgan-ex-1p-real.png) |
| :----------------------------------------------------------: |
|                        *Real dataset*                        |

<br />

**Training**

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-10e.png) | ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-25e.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         *10 epochs*                          |                         *25 epochs*                          |
| ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-40e.png) | ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-60e.png) |
|                         *40 epochs*                          |                         *60 epochs*                          |
| ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-100e.png) | ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-200e.png) |
|                         *100 epochs*                         |                         *200 epochs*                         |

<br />

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-gloss.png) | ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-closs.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                 *Generator Loss over epochs*                 |                  *Critic Loss over epochs*                   |

<br />

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/wgan-ex-1p-fid.png) |
| :----------------------------------------------------------: |
|                 *Frechet Inception distance*                 |

<br />

<br />

#### 4) Training example

| ![wgan-10e.png]({{site.baseurl}}/images/gans/wgan-ex-e10.png) | ![wgan-50e.png]({{site.baseurl}}/images/gans/wgan-ex-e50.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                         *10 epochs*                          |                         *50 epochs*                          |
| ![wgan-100e.png]({{site.baseurl}}/images/gans/wgan-ex-e100.png) | ![wgan-250e.png]({{site.baseurl}}/images/gans/wgan-ex-e250.png) |
|                         *100 epochs*                         |                         *250 epochs*                         |
| ![wgan-400e.png]({{site.baseurl}}/images/gans/wgan-ex-e400.png) | ![wgan-600e.png]({{site.baseurl}}/images/gans/wgan-ex-e600.png) |
|                         *400 epochs*                         |                         *600 epochs*                         |

| ![wgan-800e.png]({{site.baseurl}}/images/gans/wgan-ex-e800.png) |
| :----------------------------------------------------------: |
|                         *800 epochs*                         |

<br />

| ![dcgan-discri.svg]({{site.baseurl}}/images/gans/wgan-ex-closs.png) | ![dcgan-gen.svg]({{site.baseurl}}/images/gans/wgan-ex-gloss.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|              *Discriminator Loss over updates*               |                *Generator Loss over updates*                 |

| ![dcgan-fid.png]({{site.baseurl}}/images/gans/wgan-ex-fid.png) |
| :----------------------------------------------------------: |
|           *Frechet Inception Distance over epochs*           |

<br /><br />



## C) Auxiliary Classifier GANs

ACGANs are a type of GAN where you add a label classification module to improve GANs understanding on the real data by stabilizing training. This also allows to generate label-based fake samples. It is an improvement from the Conditional GANs architecture. 

The discriminator seeks to maximize the probability of correctly classifying real and fake images and correctly predicting the class label of a real or fake image. The generator seeks  to minimize the ability of the discriminator to discriminate real and fake images whilst also maximizing the ability of the discriminator predicting the class label of real and fake images. (@machinelearningmastery)

All these advantages have a cost since it requires labels. In our case we use 2 different label type: 

- Pokemon Type (water, fire, grass, electric etc...). 18 unique types.
- Pokemon Body Type (quadruped, bipedal, wings, serpentine etc..). 10 unique body types. 

| ![acgan-schema.png]({{site.baseurl}}/images/gans/acgan-schema.png) |
| :----------------------------------------------------------: |
|                    *ACGANs global schema*                    |

<br />

This architecture is compatible with most of GANs' ones, we will describe it on a DCGANs for simplification purpose. 

In the ACGANs, every generated sample (fake) receive a label in addition to the noise. This label adds information that helps the model to create sample based on class. On the other hand, the discriminator tries to predict a label for both real and fake examples. 

Depending on the GANs architecture you compute the associate loss in which you add the auxiliary loss. Typically, we use the negative log likelihood loss since we perform a multi-class prediction.  

<br />

#### **1) Generator**

| ![acgan-discri.png]({{site.baseurl}}/images/gans/acgan-gen.png) |
| :----------------------------------------------------------: |
|                  *ACGANs Generator schema*                   |

<br />

Here is an example that shows how the label affects the generation. I use here the same noise vector with different labels (*grass, water, fire*). It has a grasp on the main color associated with the types.

| ![acgan-type-comparison.png]({{site.baseurl}}/images/gans/acgan-type-comparison.png) | ![acgan-type-comparison2.png]({{site.baseurl}}/images/gans/acgan-type-comparison2.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|              *Conditional generation example 1*              |              *Conditional generation example 2*              |

<br />

#### **2) Discriminator**

| ![acgan-discri.png]({{site.baseurl}}/images/gans/acgan-discri.png) |
| :----------------------------------------------------------: |
|                *ACGANs Discriminator schema*                 |

<br />

To evaluate the **Discriminator** we calculate its accuracy on both auxiliary and adversarial task. Also I add the auxiliary distribution entropy to track failure. 

<br />

#### **Unit test**

<br />

In order to test the full flow here (main task + auxiliary task) I need 2 samples, with 2 different labels. We will be working on this dataset:

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-real.png) |
| :----------------------------------------------------------: |
|                        *Real dataset*                        |

<br />

I experienced more instability on this architecture that I did on every others. Once the **Generator** starts to really improve, on fake/real and auxiliary tasks, it usually diverges, producing noisy images. Though, many courses claim that ACGANs are more stable than any other Conditional GANs. 

<br />

#### **Training**

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-25e.png) | ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-150e.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          25 epochs                           |                         *150 epochs*                         |
| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-200e.png) | ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-280e.png) |
|                          200 epochs                          |                         *280 epochs*                         |

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-350e.png) |
| :----------------------------------------------------------: |
|                          350 epochs                          |

<br />

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-advacc.png) |
| :----------------------------------------------------------: |
|            *Accuracy on main discriminator task*             |

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-fauxacc.png) | ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-rauxacc.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|  *Accuracy on auxiliary discriminator task for fake sample*  |   Accuracy on auxiliary discriminator task for real sample   |

| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-loss.png) |
| :----------------------------------------------------------: |
|        *Generator and Discriminator Loss over epochs*        |
| ![DCGAN generator.png]({{site.baseurl}}/images/gans/acgans-1p-fid.png) |
|                 *Frechet Inception distance*                 |

<br />

The training is not stable though. The model tends to diverge once it finds an optima. It is clearly noticeable on loss and FID curves around 350 epochs. I'm currently working on it. 

<br />

<br />

### Training example





<br /><br />

## D) Architecture & Hyper Parameters

<br />

#### **1) Hyper parameters** 

Here I talk about the hyper parameters, and regularization techniques I implemented that stabilizes training, avoiding mode collapse or divergence.

<br />

- **Optimizer & learning rate**

The learning rate is often seen as a major hyper parameters to be tuned, and I cannot disagree on this one. A small modification could heavily affects the learning process, in terms of speed and convergence. To obtains decent results I usually had to set it between `1e-3` and `1e-5` . Setting different learning rate for each network (D and G) is totally okay but I did not witness any patterns or rules. Generally, if a network outperform the other you can try to speed up its learning by increasing it, but it can also affect convergence. 

Concerning the optimizer algorithm, I tried all *Stochastic Gradient Descent, RMSProp,* and *Adam*. They all worked at different settings but I experienced more divergence scenario with *SGD* and *RMSProp*. Of course it mainly depends on the task to solve so I cannot really recommend any for a GANs project. In my case I chose *Adam* with a momentum ($$\beta_1$$) set to `0.5`.

Below I show some intuitive example that shows how the learning rate affects GANs training.

| ![lr-comparison.png]({{site.baseurl}}/images/gans/lr-comparison-auxfacc.png) | ![lr-comparison.png]({{site.baseurl}}/images/gans/lr-comparison-auxracc.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|            *Auxiliary accuracy on fake examples*             |            *Auxiliary accuracy on real examples*             |

| ![lr-comparison.png]({{site.baseurl}}/images/gans/lr-comparison-genloss.png) |
| :----------------------------------------------------------: |
|                       *Generator Loss*                       |

The orange one has a Generator learning rate of `1e-2` while the blue one has it set to `5e-4`.  The auxiliary learning for generated images seems to be faster in the first 50 epochs but then rapidly diverges, on the other hand the auxiliary precision on real sample seems to be more stable. The issue comes from the Generator, diverging and producing noisy images. This is easily observable with the generator loss, it has more variance and usually reaches high peaks that last for severals epochs (producing noisy images and having difficulties to recover from it). This *Generator* issues makes the whole learning process instable. 

<br />

- **Neural network features**

The number of features, (i.e features map) are the number of kernel learned at each convolutional block. The more you put in the network, the more complex structure you can learn, but the more computational power it needs. 

| ![feature-racc.png]({{site.baseurl}}/images/gans/feature-racc.png) | ![feature-racc.png]({{site.baseurl}}/images/gans/feature_facc.png) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|           *Discriminator accuracy on real samples*           |           *Discriminator accuracy on fake samples*           |
| ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/feature-dloss.png) | ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/feature-gloss.png) |
|                     *Discriminator loss*                     |                       *Generator loss*                       |
| ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/feature-fid.png) | ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/feature-legend.png) |
|                 *Frechet Inception Distance*                 |                           *legend*                           |

It's not clear if there is a certain trend to see on this above curves. This is reassuring though, for this analysis the number of feature is shared across both networks. Then, each is competing with a network as deep as their own and it is complicated to fool the other one. The interesting part to see is the generated images. 

<br />

<br />

| ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/feature-fake-1.png) |
| :----------------------------------------------------------: |
| ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/feature-fake-2.png) |
|               *Fake samples around epoch 100*                |

We can clearly see here the shape and texture comparison. Low feature model tends to produce more trivial shapes and uniform texture, while high number of feature generates more complex structures. 

<br />

<br />

Concerning computational power needed, below you will find a comparison table, in terms of second necessary per epoch.

| Number of features in both network | Time per epoch |
| :--------------------------------: | :------------: |
|                 4                  |      3.97      |
|                 8                  |      4.77      |
|                 16                 |      4.99      |
|                 32                 |      5.05      |
|                 64                 |      7.34      |
|                128                 |     18.21      |

<br />

- **Label smoothing**

In classification tasks, label smoothing is regularization techniques that aims at preventing overconfidence and thus helps your model to generalize (being accurate on unseen data). Instead of having `0` and `1` class, we feed our loss function with smoothed labels (`0.9, 0.8` and `0.1, 0.2` etc..). In the context of GANs, we improve our discriminator network ability to generalize with this trick. In [this paper](https://arxiv.org/pdf/1606.03498.pdf), they advise to only smooth the real labels, therefore we only replace the real samples labels with `0.9`. 

<br />

| ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/label-smoothing-facc.png) |
| :----------------------------------------------------------: |
|           *Discriminator accuracy on fake example*           |

The label smoothing techniques allow our discriminator to perform better on fake examples even though the smoothing is only applied on real samples. Generalizing better enables the discriminator to give relevant feedback to the generator network and thus improves the overall model.

<br />

This method is well fitted for most of GANs architecture but not all of them. It needs to have a classifier in order to smooth labels, WGANs does not use it and thus cannot benefit from label smoothing. This is why I used another regularization technique for the Wasserstein GANs, I described it hereafter.  

<br />

- **Instance noise**

Adding noise to input is often what you want to make your network more robust (*ie Dropout, Layers Noise etc..*). The whole point of this method is to make the discriminator job harder to obtain convergence. I incite you to read [this article](https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/) that explains the whole idea precisely. The technique involves adding Gaussian noise to the discriminator input (*ie fake and real samples*). 

<br />

- **Z space dimension and distribution**

The $$Z$$ space is where you sample all the inputs you feed to the Generator, it can be significant in the quality of generated images, especially in the diversity aspect. 

So I tried to compare two ways of sampling, uniform distribution ($$\mathcal{U}(-1, 1)$$ and Normal distribution. As far as I studied, I did not notice major differences between the 2's. Most of the time, the results were equal or not significantly better.

<br />

| ![label-smoothing-facc.png]({{site.baseurl}}/images/gans/z-fid.png) |
| :----------------------------------------------------------: |
|                 *Frechet Inception distance*                 |

Above is the FID for the same network trained with Z samples from 2 different distributions.

<br /><br />

- **Neural network weight initialization**

Here we will compare our networks efficiency with different weights initializations. Deep neural nets can easily suffer from vanishing or exploding gradient during training, especially with GANs whose training lacks stability. A good weights initialization can free us from these issues, [this article](https://www.deeplearning.ai/ai-notes/initialization/) shows it nicely. As I am using only `Linear` and `Conv2d` layers from pytorch lib, the default initialization is the He initialization. That is, basically, a continuous uniform distribution between 2 values based on the network size (input features for `Linear` and input channels/ kernel size for `Conv2d`). The original GANs paper recommends to initialize with a Gaussian distribution ($$\mathcal{N_{0, 0.02}}$$). 

<br />

On this particular task, I tried to observe empirically benefits between the He initialization and the  recommended Gaussian but I did not witness anyone performing better than the other. It must makes differences in the early training for some cases, depending on other parameters. 

<br />

<br />

#### 2) Generator upsampling methods

Convolutional neural networks are originally built to take an image as input, in the Generator case we want to do it the other way. To do so, we have several options. We will explore 3 different methods:

- **Transpose Convolution**

It is the operation inverse to convolution. 

![convtranspose.gif]({{site.baseurl}}/images/gans/convtranspose.gif)

*<br />*

- **Convolution and Upsample (nearest)**

*![upsampling.png]({{site.baseurl}}/images/gans/upsampling.png)*



*<br />*

- **Color Picker** 

Color Picker is a technique I found [here](https://github.com/ConorLazarou/PokeGAN), the idea is to generate separately each Red Green Blue channel.  Each is created thanks to 2 components, a color palette and a distribution over this palette, we use the latter to weight the palette and to create a single channel 64x64 matrix. The palette tensor is the same for each channel while the distribution is computed 3 times (R,G,B). 

<br />

**Comparison**

![comparison-upsample-convtranspose.png]({{site.baseurl}}/images/gans/comparison-upsample-convtranspose.png)

Here we have fake samples from a WGAN, with both upsample + conv method and convtranspose. The Upsample method seems to create more complex structure. It is a benefit when talking about Pokemon shape since it outputs limbs etc..  but a drawback concerning colors. The generated image has too many colors making the Pokemon unreal. Even with a long training, it appears that this architecture has a hard time creating a uniform color shape. 

The ColorPicker Generator architecture answers this handicap. Indeed it benefits from the complexity given by the upsample method and gets uniform colors with the help of the palette-oriented architecture. 

| *![colorpicker-ex.png]({{site.baseurl}}/images/gans/colorpicker-ex.png)* |
| :----------------------------------------------------------: |
|             *Color Picker architecture example*              |

<br /><br />

| *![gan-meme.png]({{site.baseurl}}/images/gans/gan-meme.png)* |
| :----------------------------------------------------------: |
|              *Thank for reaching the end of it*              |





