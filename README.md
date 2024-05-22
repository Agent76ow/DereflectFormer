# VDereflectFormer: Vision Transformers for Single Image Reflection Removal

> **Abstract:** 
We address the challenge of single image reflection re-
moval (SIRR), a crucial task in computer vision that involves eliminating unde-
sirable reflections from images captured through glass surfaces. Current state-
of-the-art methods typically rely on convolutional neural networks (CNNs) and
often make certain assumptions about the appearance of reflections, which may
not hold true in real-world scenarios. To overcome these limitations, we pro-
pose a novel Transformer-based approach, DereflectFormer, inspired by the Swin
Transformer. Our architecture introduces a new module, the Depthwise Multi-
Activation Feed-Forward Network (DMFN), which leverages depthwise convo-
lution and a dual-stream ReLU-GELU activation function to enhance detail ex-
traction capability. We also employ a synthetic dataset and a synthesis method for
training, which allows our model to fully exploit the capabilities of Transformer
architectures. Experimental results demonstrate that our approach outperforms
state-of-the-art methods, providing more accurate and robust results in various
real-world scenarios. Furthermore, our ablation studies reveal that each compo-
nent of our architecture contributes significantly to its performance, offering valu-
able insights for future research in the field of single image reflection removal.

![DereflecFormer](figs/Dereflectformer.png)
