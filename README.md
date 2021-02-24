# PCN: Point Completion Network

![PCN](images/network.png)

This is implementation of PCN——Point Completion Network in pytorch. PCN is an autoencoder for point cloud completion. As for the details of the paper, please refer to [arXiv](https://arxiv.org/pdf/1808.00671.pdf).

## Environment

* Python 3.7.9
* PyTorch 1.7.0
* CUDA 10.1.243

## Reconstruction

### alpha = 0.1

Time | Loss
-- | --
1 | 0.1406116538370649
2 | 0.13831139862951305
3 | 0.13889332777924007
4 | 0.13953917752951384
5 | 0.14458834902486867

### alpha = 0.5
Time | Loss
-- | --
1 | 0.08169769392245345
2 | 0.08213998491151465
3 | 0.0818117022410863
4 | 0.08218497120671803
5 | 0.08180649588919348

### alpha = 1.0
Time | Loss
-- | --
1 | 0.08289035016463862
2 | 0.08557101111445162
3 | 0.08746305232246716
4 | 0.08545542767064439
5 | 0.08654008086563812

## Examples