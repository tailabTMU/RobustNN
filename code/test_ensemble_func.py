import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel, MNISTDPModel
from setup_inception import ImageNet, InceptionModel
import multiprocessing as mp
from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

def makeDir(strPath):
    if not os.path.exists(strPath):
        try:
            os.makedirs(strPath)
        except:
            print('skip makedir')
    return strPath

## original code by Nicholas Carlini
def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def plotCIFAR(img, strImageFolder='', strImageName=''):
    timg = (img+0.5)*255
    timg = np.array(timg, dtype='uint8')
    timg = timg.reshape(32,32,3)
    plt.imshow(timg)
    if strImageFolder != '':
        plt.savefig(strImageFolder+strImageName)
    else:
        plt.show()
    plt.close()

def plotMNIST(img, strImageFolder='', strImageName=''):
    timg = (img+0.5)*255
    timg = np.array(timg, dtype='uint8')
    timg = timg.reshape(28,28)
    plt.imshow(timg, cmap='gray')
    if strImageFolder != '':
        plt.savefig(strImageFolder+strImageName)
    else:
        plt.show()
    plt.close()


def showCifar(img):
    timg = (img+0.5)*255
    plt.imshow(timg)   
    plt.show()
    plt.close()

## original code by Nicholas Carlini
def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

#superimposition attack
def test_ensemble_noisylogit_superimposed_opt(data, datamodel, arr_file_name, file_name_out, file_name_summary, nAttacked, samples=1, start=0, plotIMG=plotMNIST, strImgFolder=''):
    arr_models = []
    arr_attack = []
    arr_adv = []
    f_summary = open(file_name_summary, 'w')
    f_out = open(file_name_out, 'w')
    #generate samples and targets
    inputs, targets = generate_data(data, samples=samples, targeted=True, start=start, inception=False)
    timestart = time.time()
    timeend = time.time()
    f_summary.write('Begin Timestamp: '+str(timestart)+'\n')
    f_summary.write('Test on '+str(samples)+' samples; start position = '+str(start)+'\n')
    f_out.write('Begin Timestamp: '+str(timestart)+'\n')
    f_out.write('Test on '+str(samples)+' samples; start position = '+str(start)+'\n')
    with tf.Session() as sess:
        #restore ensemble networks from files
        for file_model in arr_file_name:
            model = datamodel(file_model, sess)
            arr_models.append(model)
        #generate attacks for each network in the ensemble for each sample-target pair
        for i in range(len(arr_models)):
            attack = CarliniL2(sess, arr_models[i], batch_size=9, max_iterations=1000, confidence=0)
            arr_attack.append(attack)
            timestart = time.time()
            adv = attack.attack(inputs, targets)
            timeend = time.time()
            arr_adv.append(adv)
        for j in range(len(inputs)):
            arr_classification = []
            arr_distortion = [np.sum((adv[j:j+1]-inputs[j:j+1])**2)**.5 for adv in arr_adv]
            f_summary.write('all distortions: '+';'.join([str(dist) for dist in arr_distortion])+'\n')
            #get the attacks with the smallest distortiohs
            arr_indices = np.array(arr_distortion).argsort()[0:nAttacked]
            strAttInd = '-'.join([str(ind) for ind in arr_indices])
            arrAttackedImage = [arr_adv[ind][j:j+1] for ind in arr_indices]
            #get superimposition of the chosen attacks
            tmpImposed = get_superimposed(inputs[j:j+1], arrAttackedImage)
            strImageInput = 'a'+strAttInd+'_input'+str(j)
            print('input image: +\n')
            #save the sample image
            plotIMG(inputs[j], strImgFolder, strImageInput+'.png')
            show(inputs[j])
            f_summary.write('Target: '+str(np.argmax(targets[j]))+'\n')
            for a in range(nAttacked):
                strImageAdv = strImageInput+'_adv'+str(arr_indices[a])
                print('adv image'+str(arr_indices[a])+': \n')
                #save each attacked image in the superimposition
                plotIMG(arrAttackedImage[a], strImgFolder, strImageAdv+'.png')
                show(arrAttackedImage[a])
                adv_distortion = np.sum((arrAttackedImage[a]-inputs[j:j+1])**2)**.5
                f_summary.write(strImageAdv+' perturbation: '+str(adv_distortion)+'\n')
            print('adv image superimposed: +\n')
            strImageSup = strImageInput+'_advsup'
            #save the superimposed image
            plotIMG(tmpImposed, strImgFolder,strImageSup+'.png')
            show(tmpImposed)
            for k in range(len(arr_file_name)):
                x = tf.placeholder(tf.float32, (None, arr_models[k].image_size, arr_models[k].image_size, arr_models[k].num_channels))
                model_original = arr_models[k].predict(x)
                #classify the original input
                class_original = sess.run(model_original, {x: inputs[j:j+1]})
                #classify the superimposed image
                class_adv = sess.run(model_original, {x: tmpImposed})
                f_out.write(arr_file_name[k]+" Classification: "+str(np.argmax(class_adv))+'\n')
                f_summary.write(arr_file_name[k]+" original Classification: "+str(np.argmax(class_original))+'\n')
                f_summary.write(arr_file_name[k]+" adv Classification: "+str(np.argmax(class_adv))+'\n')
                arr_classification.append(np.argmax(class_adv))
            total_distortion = np.sum((tmpImposed-inputs[j:j+1])**2)**.5
            img_norm =  np.sum((inputs[j:j+1])**2)**.5
            #write the final vote
            f_summary.write('Final vote: '+str(np.bincount(arr_classification).argmax())+'\n')
            print("Total distortion:", total_distortion)
            print('Image norm: ', img_norm)
            f_out.write("Total distortion: "+str(total_distortion)+'\n')
            f_out.write('Image norm: '+str(img_norm)+'\n')
            f_summary.write("Total distortion: "+str(total_distortion)+'\n')
            f_summary.write('Image norm: '+str(img_norm)+'\n')
    f_summary.write('End Timestamp: '+str(timeend)+'\n')
    f_out.write('End Timestamp: '+str(timeend)+'\n')
    f_out.close()
    f_summary.close()  

#check transferability
def test_ensemble_noisylogit_superimposed_all_byinput(data, datamodel, arr_file_name, file_name_summary, samples=1, start=0, plotIMG=plotMNIST, strImgFolder=''):
    arr_models = []
    arr_attack = []
    arr_adv = []
    f_summary = open(file_name_summary, 'w')
    inputs, targets = generate_data(data, samples=samples, targeted=True, start=start, inception=False)
    timestart = time.time()
    timeend = time.time()
    f_summary.write('Begin Timestamp: '+str(timestart)+'\n')
    f_summary.write('Test on '+str(samples)+' samples; start position = '+str(start)+'\n')
    dict_class_orig = [[] for ind in range(len(arr_file_name))]
    dict_class_adv = [[] for ind in range(len(arr_file_name))]
    with tf.Session() as sess:
        #restore ensemble networks from files
        for file_model in arr_file_name:
            model = datamodel(file_model, sess)
            arr_models.append(model)
        #generate attacks for each network in the ensemble for each sample-target pair
        for i in range(len(arr_models)):
            attack = CarliniL2(sess, arr_models[i], batch_size=9, max_iterations=1000, confidence=0)
            arr_attack.append(attack)
            timestart = time.time()
            adv = attack.attack(inputs, targets)
            timeend = time.time()
            arr_adv.append(adv)
        print('attacks done.')
        get_all_cpu_mem_info()
        pool = mp.Pool(mp.cpu_count())
        arr_target = [str(np.argmax(targets[j])) for j in range(len(inputs))]
        arr_input = [str(j) for j in range(len(inputs))]
        arr_img_input = ['input'+arr_input[j]+'_target'+arr_target[j] for j in range(len(inputs))]
        #save each input image
        imgs_input_results = [pool.apply(plotIMG, args=(inputs[j],strImgFolder,arr_img_input[j]+'.png')) for j in range(len(inputs))]
        arr_feeds = [inputs[j:j+1] for j in range(len(inputs))]
        print('starting quries')
        get_all_cpu_mem_info()
        for ind in range(len(arr_models)):
            arrAttackedImage = [arr_adv[ind][j:j+1] for j in range(len(inputs))]
            x = tf.placeholder(tf.float32, (None, arr_models[ind].image_size, arr_models[ind].image_size, arr_models[ind].num_channels))
            #save each attacked image
            arr_img_adv = [arr_img_input[j]+'_adv'+str(ind) for j in range(len(inputs))]
            imgs_adv_results = [pool.apply(plotIMG, args=(arrAttackedImage[j],strImgFolder,arr_img_adv[j]+'.png')) for j in range(len(inputs))]
            arr_class_orig_byind = []
            arr_class_adv_byind = []
            for k in range(len(arr_models)):
                #for each network, classify all original inputs
                arr_class_orig_byind.append(get_prediction_array(sess, arr_models[k], x, list(inputs)))
                #for each network, classify all attacked images targeted at a specific network
                arr_class_adv_byind.append(get_prediction_array(sess, arr_models[k], x, list(arr_adv[ind])))
            dict_class_orig[ind] = arr_class_orig_byind
            dict_class_adv[ind] = arr_class_adv_byind
            get_all_cpu_mem_info()
    print('starting write output')
    get_all_cpu_mem_info()
    for j in range(len(inputs)):
        arr_classification = []
        arr_distortion = [np.sum((adv[j:j+1]-inputs[j:j+1])**2)**.5 for adv in arr_adv]
        str_target = str(np.argmax(targets[j]))
        f_summary.write('Target: '+str_target+'\n')
        f_summary.write('all distortions: '+';'.join([str(dist) for dist in arr_distortion])+'\n')
        img_norm = np.sum((inputs[j:j+1])**2)**.5
        f_summary.write('Image norm: '+str(img_norm)+'\n')
        for ind in range(len(arr_models)):
            strImageAdv = arr_img_input[j]+'_adv'+str(ind)
            out_class_orig = [dict_class_orig[ind][k][j] for k in range(len(arr_models))]
            out_class_adv = [dict_class_adv[ind][k][j] for k in range(len(arr_models))]
            str_class_orig = ','.join([str(label) for label in out_class_orig])
            str_class_adv = ','.join([str(label) for label in out_class_adv])
            f_summary.write(strImageAdv+",original Classification: "+str_class_orig+'\n')
            f_summary.write(strImageAdv+",adv Classification: "+str_class_adv+'\n')
            f_summary.write(strImageAdv+',original vote: '+str(np.bincount(out_class_orig).argmax())+'\n')
            f_summary.write(strImageAdv+',adv vote: '+str(np.bincount(out_class_adv).argmax())+'\n')
    f_summary.write('End Timestamp: '+str(timeend)+'\n')
    f_summary.close()



def get_prediction_array(sess, model, x, feeds):
    pred_model = model.predict(x)
    preds = sess.run(pred_model, {x:feeds})
    return [np.argmax(pred) for pred in preds]


def get_superimposed(baseImage, arrAdvImage, tol=1e-10):
    arr_diff = []
    tmpImposed = baseImage
    for advImage in arrAdvImage:
        arr_diff.append(advImage-baseImage)
    for diff in arr_diff:
        tmpImposed = tmpImposed + diff
    tmpImposed[tmpImposed>0.5] = 0.5
    tmpImposed[tmpImposed<-0.5] = -0.5
    return tmpImposed

