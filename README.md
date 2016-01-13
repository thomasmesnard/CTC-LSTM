CTC-LSTM
=========================================

This repositry contains an implementation of the CTC cost function (Graves et al., 2006).

- CTC cost is implemented in pure [Theano](https://github.com/Theano/Theano).

- Supports mini-batch.

To avoid numerical underflow, two solutions are implemented:

- Normalization of the alphas at each timestep
- Calculations in the logarithmic domain

This repository also contains sample code for applying CTC to two datasets, a simple dummy dataset constituted of artificial data, and code to use the TIMIT dataset. The models are implemented using [Blocks](https://github.com/mila-udem/blocks). Both datasets are implemented using [Fuel](https://github.com/mila-udem/fuel).

The model on the TIMIT dataset is able to learn up to 50% phoneme accuracy using no handcrafted processing of the signal, but instead uses an end-to-end model composed of convolutions, LSTMs, and the CTC cost function. 


Reference
=========
Graves, Alex, et al. *Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks.* Proceedings of the 23rd international conference on Machine learning. ACM, 2006.


Credits
=======
[Alex Auvolat](https://github.com/Alexis211)

[Thomas Mesnard](https://github.com/thomasmesnard)


Special thanks to
=================
[Mohammad Pezeshki](https://github.com/mohammadpz/)
