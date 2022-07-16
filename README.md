# A Semismooth Newton based Augmented Lagrangian Method for Nonsmooth Optimization on Matrix Manifolds

Yuhao Zhou, Chenglong Bao, Chao Ding, Jun Zhu. 
_Preprint. [Arxiv:2103.02855](https://arxiv.org/abs/2103.02855)._

## Usage

Before running our code, please load the package `manopt` first in MATLAB:

```matlab
run manopt/importmanopt.m
```

Then, you can run `RunCM.m`, `RunSPCA.m` or `RunOrtho.m`.

_Important note_: this is a research software. It is not intended nor designed to be a general purpose software at the moment.

## Files 

Our codes are in the directory `almssn`. They depend on two manifold optimization libraries [1, 2]:

 - The directory `manopt` contains the package http://www.manopt.org [1].
 - The directory `OptM` contains Wen's first-order method in Stiefel manifolds; see https://github.com/optsuite/OptM [2]. The file `OptStiefelGBB.m` was slightly modified to record the intermediate results.

## References

 - [1]: N. Boumal, B. Mishra, P.-A. Absil, and R. Sepulchre, Manopt, a Matlab toolbox for optimization on manifolds, Journal of Machine Learning Research, 15 (2014), pp. 1455–1459.
 - [2]: Z. Wen and W. Yin, A feasible method for optimization with orthogonality constraints, Mathematical Programming, 142 (2013), pp. 397–434.
