from train_models_ensemble import train_teachers_all
from setup_mnist import MNIST
from setup_cifar import CIFAR

mnist_params = [32, 32, 64, 64, 200, 200]
cifar_params = [64, 64, 128, 128, 256, 256]
arr_temp = [i*10 for i in range(1,51)]

# train ensemble of networks for MNIST with partitioned training data
train_dir = "models_ensemble_mnist_T10-500/mnist_3000epoch_"
train_teachers_all(MNIST(),mnist_params,arr_temp,train_dir,arrInit=None,num_epochs=3000)


# train ensemble of networks for CIFAR with with non-partitioned training data
train_dir = "models_ensemble_cifar_T10-500/cifar_150epoch_"
train_teachers_all(CIFAR(),cifar_params,arr_temp,train_dir,arrInit=None,num_epochs=150,bool_partition=False)
