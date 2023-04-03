# Spatial Transformation Network (STN)

- Assist in detecting most important ROI
- Then, determine what transformation is necessary

The Spatial Transformer Module:
1. Localization Network
    - take the input feature map ${\bold U}$ of W width, H height, and C number of channels
    - output ${\Theta}$, the transformation parameters to be applied
    - *must contain a final regression layer to compute ${\Theta}$*
2. Grid Generator
    - input feature map ${\bold U}$ and transformation parameters ${\Theta}$
    - output feature map ${\bold V}$ is a square grid
    - **Transformation Grid** gives: ${\bold U} = T_{\Theta}(G_i) = A_{\Theta} \times {\bold V}$
        - meaning, for each source coordinate (from $\bold U$), it is defined as the transformation matrix ($A_{\Theta}$) multiplied by each target coordinate in $\bold V$
3. Sampler
    - The output feature map $\bold V$ values will be estimated using input pixel values
        - use linear/bilinear interpolation of output from input pixels



### File Structure:

```
.
└── spatial-transformation/
    ├── config.py
    ├── STN.py
    ├── classifier.py
    ├── callback.py
    ├── train.py
    └── info.md
```





### CITATIONS:
Chakraborty, D. “Spatial Transformer Networks Using TensorFlow,” PyImageSearch, P. Chugh, A. R. Gosthipaty, S. Huot, K. Kidriavsteva, R. Raha, and A. Thanki, eds., 2022, https://pyimg.co/4ham6

https://tree.nathanfriend.io