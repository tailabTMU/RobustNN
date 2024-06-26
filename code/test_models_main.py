from test_ensemble_func import test_ensemble_noisylogit_superimposed_opt, plotCIFAR, plotMNIST, makeDir, test_ensemble_noisylogit_superimposed_all_byinput
from setup_cifar import CIFAR, CIFARModel, CIFARDPModel
from setup_mnist import MNIST, MNISTModel, MNISTDPModel


#dataset for the tests
data = MNIST()#CIFAR()
#model for the networks
model = MNISTModel#CIFARModel#MNISTDPModel#CIFARDPModel
#start position of sample
nStart = 0 
#number of samples
nSamples = 10
#number of samples to generate at a time
nBatch = 1
#plotting function for saving images
plot_img = plotMNIST
#directory containing the ensemble of networks
strFolder = 'models_ensemble_mnist_T10-500/'
#filename prefix of the trained networks
strFilePrefix = 'mnist_3000epoch_50_teachers_'
#array of filenames for the trained networks
arrFilenames = []
for i in range(50):
    arrFilenames.append(strFolder+ strFilePrefix + str(i))


# test transferability
for iStart in range(nStart,nStart+nSamples,nBatch):
    folder_image_save = strFolder+'test_img_all50model_'+str(iStart+1)+'-'+str(iStart+nBatch)+'/'
    folder_image_save = makeDir(folder_image_save)
    file_output_summary = folder_image_save+'testoutput_'+str(iStart+1)+'-'+str(iStart+nBatch)+'_summary.txt'
    test_ensemble_noisylogit_superimposed_all_byinput(data, model, arrFilenames, file_output_summary, samples=nBatch, start=iStart, plotIMG=plot_img, strImgFolder=folder_image_save)


# test superimposition of 2 or 3
nSup = 2 #3
for j in range(nStart,nStart+nSamples,nBatch):
    folder_image_save = strFolder+'sup'+str(nSup)+'_testmnist_'+str(j+1)+'-'+str(j+nBatch)+'/'
    folder_image_save = makeDir(folder_image_save)
    file_output = folder_image_save+'testoutput_'+str(j+1)+'-'+str(j+nBatch)+'.txt'
    file_output_summary = folder_image_save+'testoutput_'+str(j+1)+'-'+str(j+nBatch)+'_summary.txt'
    test_ensemble_noisylogit_superimposed_opt(data, model, arrFilenames, file_output, file_output_summary, nSup, samples=nBatch, start=j, plotIMG=plot_img, strImgFolder=folder_image_save)
