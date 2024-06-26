import numpy as np
import glob

def process_all_results(strFolder, bSupImp=False):
    arrFilenames = glob.glob(strFolder+'*summary*.txt')
    f = open(strFolder+'master.txt','w')
    extract_method = extract_results
    if bSupImp:
        extract_method = extract_results_supimposed
    strHeader, arrResults = extract_method(arrFilenames[0])
    f.write(strHeader+'\n')
    for resLine in arrResults:
        f.write(resLine+'\n')
    for i in range(len(arrFilenames)-1):
        strHeader, arrResults = extract_method(arrFilenames[i+1])
        for resLine in arrResults:
            f.write(resLine+'\n')
    f.close()

def extract_results(strFilename):
    nSamples = 0
    iSample = 0
    iTarget = 0
    bContainDistortion =  False
    dictTarget2Class = {}
    dictSample2Target = {}
    arrTargets = [i for i in range(10)]
    with open(strFilename) as f:
        dictClass = {}
        target = 0
        fvote = 0
        distortion = 0
        for line in f:
            if 'Test on' in line:
                nSamples = int(line.split(' ')[2])
            elif 'Target' in line:
                target = int(line.split(' ')[1])
                arrTargets.remove(target)
            elif 'Classification:' in line:
                tmpSplit = line.split(': ')
                strKey = tmpSplit[0].lower().split('-')[-1]
                dictClass[strKey] = int(tmpSplit[1])
            elif 'Final vote' in line:
                fvote = int(line.split(': ')[1])
                dictTarget2Class[target] = dictClass
                dictTarget2Class[target]['final vote'] = fvote
                dictClass = {}
                iTarget = iTarget + 1
            elif 'Total distortion' in line:
                distortion = float(line.split(': ')[1])
                dictTarget2Class[target]['total distortion'] = distortion
            if iTarget >= 9 and ('/' in line or 'End Timestamp' in line):
                for t in dictTarget2Class.keys():
                    dictTarget2Class[t]['input'] = arrTargets[0]
                dictSample2Target[iSample] = dictTarget2Class
                iTarget = 0
                iSample = iSample + 1
                arrTargets = [i for i in range(10)]
                dictTarget2Class = {}
    arrResults = []
    arrHeaders = ['input']
    for i, sample in dictSample2Target.items():
        strResult = ''
        if len(arrHeaders) < 2:
            tar = list(sample.keys())[0]
            for key in sample[tar].keys():
                if 'classification' in key:
                    arrHeaders.append(key)
            arrHeaders.append('final vote')
            if 'total distortion' in sample[tar].keys():
                arrHeaders.append('total distortion')
        for target, res in sample.items():
            strResult = strFilename+','+str(i)+','+str(target)+','+','.join([str(res[name]) for name in arrHeaders])
            arrResults.append(strResult)
    strHeader = 'filename,sample,target,'+','.join(arrHeaders)
    return strHeader, arrResults
            
def extract_results_supimposed(strFilename):
    nSamples = 0
    iSample = 0
    iTarget = 0
    bContainDistortion =  False
    dictTarget2Class = {}
    dictSample2Target = {}
    arrTargets = [i for i in range(10)]
    with open(strFilename) as f:
        dictClass = {}
        target = 0
        fvote = 0
        distortion = 0
        strDistortAll = ''
        for line in f:
            if 'Test on' in line:
                nSamples = int(line.split(' ')[2])
            elif 'Target' in line:
                target = int(line.split(' ')[1])
                arrTargets.remove(target)
            elif 'Classification:' in line:
                tmpSplit = line.split(': ')
                strKey = tmpSplit[0].lower().split('-')[-1]
                dictClass[strKey] = int(tmpSplit[1])
            elif 'Final vote' in line:
                fvote = int(line.split(': ')[1])
                dictTarget2Class[target] = dictClass
                dictTarget2Class[target]['final vote'] = fvote
                dictClass = {}
                iTarget = iTarget + 1
            elif 'Total distortion' in line:
                distortion = float(line.split(': ')[1])
                dictTarget2Class[target]['total distortion'] = distortion
            elif 'Image norm' in line:
                norm = float(line.split(': ')[1])
                dictTarget2Class[target]['image norm'] = norm
                dictTarget2Class[target]['all distortions'] = strDistortAll
            elif 'all distortions' in line:
                strDistortAll = line.split(': ')[1][:-1]
            if iTarget >= 9 and 'Image norm' in line:
                for t in dictTarget2Class.keys():
                    dictTarget2Class[t]['input'] = arrTargets[0]
                dictSample2Target[iSample] = dictTarget2Class
                iTarget = 0
                iSample = iSample + 1
                arrTargets = [i for i in range(10)]
                dictTarget2Class = {}    
    arrResults = []
    arrHeaders = ['input']
    for i, sample in dictSample2Target.items():
        strResult = ''
        if len(arrHeaders) < 2:
            tar = list(sample.keys())[0]
            for key in sample[tar].keys():
                if 'classification' in key:
                    arrHeaders.append(key)
            arrHeaders.append('final vote')
            if 'total distortion' in sample[tar].keys():
                arrHeaders.append('total distortion')
            if 'image norm' in sample[tar].keys():
                arrHeaders.append('image norm')
            if 'all distortions' in sample[tar].keys():
                arrHeaders.append('all distortions')
        for target, res in sample.items():
            strResult = strFilename+','+str(i)+','+str(target)+','+','.join([str(res[name]) for name in arrHeaders])
            arrResults.append(strResult)
    strHeader = 'filename,sample,target,'+','.join(arrHeaders)
    return strHeader, arrResults

def extract_results_supimposed_dict(strFilename):
    nSamples = 0
    iSample = 0
    iTarget = 0
    bContainDistortion =  False
    dictTarget2Class = {}
    dictSample2Target = {}
    arrTargets = [i for i in range(10)]
    with open(strFilename) as f:
        dictClass = {}
        target = 0
        fvote = 0
        distortion = 0
        strDistortAll = ''
        for line in f:
            if 'Test on' in line:
                nSamples = int(line.split(' ')[2])
            elif 'Target' in line:
                target = int(line.split(' ')[1])
                arrTargets.remove(target)
            elif 'Classification:' in line:
                tmpSplit = line.split(': ')
                strKey = tmpSplit[0].lower().split('-')[-1]
                dictClass[strKey] = int(tmpSplit[1])
            elif 'Final vote' in line:
                fvote = int(line.split(': ')[1])
                dictTarget2Class[target] = dictClass
                dictTarget2Class[target]['final vote'] = fvote
                dictClass = {}
                iTarget = iTarget + 1
            elif 'Total distortion' in line:
                distortion = float(line.split(': ')[1])
                dictTarget2Class[target]['total distortion'] = distortion
            elif 'Image norm' in line:
                norm = float(line.split(': ')[1])
                dictTarget2Class[target]['image norm'] = norm
                dictTarget2Class[target]['all distortions'] = strDistortAll
            elif 'all distortions' in line:
                strDistortAll = line.split(': ')[1][:-1]
            if iTarget >= 9 and 'Image norm' in line:
                for t in dictTarget2Class.keys():
                    dictTarget2Class[t]['input'] = arrTargets[0]
                dictSample2Target[iSample] = dictTarget2Class
                iTarget = 0
                iSample = iSample + 1
                arrTargets = [i for i in range(10)]
                dictTarget2Class = {}
    return dictSample2Target

def extract_stats(strMaster, arrTeachers):
    data = np.genfromtxt(strMaster, dtype=None, delimiter=',', names=True)
    column_input = data['input']
    column_target = data['target']
    column_vote = data['final_vote']
    column_distortion = data['total_distortion']
    column_norm = data['image_norm']
    nSamples = len(column_input)
    arr_orig_accuracy = []
    arr_adv_accuracy = []
    print('teacher,original accuracy,adv accuracy')
    for strTeacher in arrTeachers:
        str_orig_class = strTeacher+'_original_classification'
        str_adv_class = strTeacher+'_adv_classification'
        column_teacher_orig = data[str_orig_class]
        column_teacher_adv = data[str_adv_class]
        count_orig = sum(column_input[i]==column_teacher_orig[i] for i in range(nSamples))
        count_adv = sum(column_input[i]==column_teacher_adv[i] for i in range(nSamples))
        arr_orig_accuracy.append(count_orig)
        arr_adv_accuracy.append(count_adv)
        print(strTeacher+','+str(count_orig)+','+str(count_adv))
    print('student,output=input,output=target,output=other')
    arr_distortion = column_distortion/column_norm
    count_input = 0
    count_target = 0
    count_other = 0
    distort_input = 0.0
    distort_target = 0.0
    distort_other = 0.0
    for i in range(nSamples):
        if column_input[i]==column_vote[i]:
            count_input = count_input + 1
            distort_input = distort_input + arr_distortion[i]
        elif column_target[i]==column_vote[i]:
            count_target = count_target + 1
            distort_target = distort_target + arr_distortion[i]
        else:
            count_other = count_other + 1
            distort_other = distort_other + + arr_distortion[i]
    if count_input > 0: distort_input = distort_input/count_input
    if count_target > 0: distort_target = distort_target/count_target
    if count_other > 0: distort_other = distort_other/count_other
    print('count,'+str(count_input)+','+str(count_target)+','+str(count_other))
    print('average distortion,'+str(distort_input)+','+str(distort_target)+','+str(distort_other))

def extract_details(strMaster, arrTeachers):
    data = np.genfromtxt(strMaster, dtype=None, delimiter=',', names=True)
    column_input = data['input']
    column_target = data['target']
    column_vote = data['final_vote']
    column_distortion = data['total_distortion']
    column_norm = data['image_norm']
    nSamples = len(column_input)
    arr_orig_accuracy = []
    arr_adv_accuracy = []
    arr_teacher_orig = []
    arr_teacher_adv = []
    arr_correct = []
    arr_target = []
    arr_other = []
    for strTeacher in arrTeachers:
        str_orig_class = strTeacher+'_original_classification'
        str_adv_class = strTeacher+'_adv_classification'
        column_teacher_orig = data[str_orig_class]
        column_teacher_adv = data[str_adv_class]
        arr_teacher_orig.append(column_teacher_orig)
        arr_teacher_adv.append(column_teacher_adv)
    arr_distortion = column_distortion/column_norm
    print(',output=input,output=target,output=other')
    for i in range(nSamples): 
        count_input = 0
        count_target = 0
        count_other = 0
        distort_input = 0.0
        distort_target = 0.0
        distort_other = 0.0
        for j in range(len(arrTeachers)):
            if column_input[i]==arr_teacher_adv[j][i]:
                count_input = count_input + 1
            elif column_target[i]==arr_teacher_adv[j][i]:
                count_target = count_target + 1
            else:
                count_other = count_other + 1
        arr_correct.append(count_input)
        arr_target.append(count_target)
        arr_other.append(count_other)
        print(str(i)+','+str(count_input)+','+str(count_target)+','+str(count_other))


def get_final_vote(strMaster, arrTeachers,strFileout=''):
    data = np.genfromtxt(strMaster, dtype=None, delimiter=',', names=True)
    column_sample = data['sample']
    column_input = data['input']
    column_target = data['target']
    column_distortion = data['total_distortion']
    column_norm = data['image_norm']
    column_alldistort  = data['all_distortions']
    nSamples = len(column_input)
    arr_orig_accuracy = []
    arr_adv_accuracy = []
    arr_teacher_orig = []
    arr_teacher_adv = []
    arr_correct = []
    arr_target = []
    arr_other = []
    f_out = None
    if strFileout != '':
        f_out = open(strFileout,'w')
    for strTeacher in arrTeachers:
        str_orig_class = strTeacher+'_original_classification'
        str_adv_class = strTeacher+'_adv_classification'
        column_teacher_orig = data[str_orig_class]
        column_teacher_adv = data[str_adv_class]
        arr_teacher_orig.append(column_teacher_orig)
        arr_teacher_adv.append(column_teacher_adv)
    arr_distortion = column_distortion/column_norm
    strheader = 'sample,target,input,original vote,noisy vote,total_distortion,image_norm,all distortions'
    if f_out != None:
        f_out.write(strheader+'\n')
    print(strheader)
    for i in range(nSamples):
        arr_orig = []
        arr_adv = []
        for j in range(len(arrTeachers)):
            arr_orig.append(arr_teacher_orig[j][i])
            arr_adv.append(arr_teacher_adv[j][i])
        str_orig = str(np.bincount(arr_orig).argmax())
        str_adv = str(np.bincount(arr_adv).argmax())
        str_out = str(column_sample[i])+','+str(column_target[i])+','+str(column_input[i])
        str_out = str_out+','+str_orig+','+str_adv+','+str(column_distortion[i])+','+str(column_norm[i])+','+str(column_alldistort[i])
        print(str_out)
        if f_out != None:
            f_out.write(str_out+'\n')
    if f_out != None:
        f_out.close()
    

def get_transfer_stats(strfilename,strFileout=''):
    fIn = open(strfilename,'r')
    strheader = 'Input,Target,Adv,Distortion,Image norm,correct to target,correct to other,orig vote,adv vote'
    strtarget = ''
    strinput = ''
    strtarget = ''
    stradv = ''
    strdistortion = ''
    strimagenorm = ''
    correct2target = 0
    correct2other = 0
    strvote_orig = ''
    strvote_adv = ''
    arr_target = [str(i) for i in range(10)]
    arr_out = []
    for line in fIn:
        if 'Target: ' in line:
            if strtarget != '':
                strtarget = ''
                strinput = ''
                strtarget = ''
                stradv = ''
                strdistortion = ''
                strimagenorm = ''
                correct2target = 0
                correct2other = 0
                strvote_orig = ''
                strvote_adv = ''
            strtarget = line[len('Target: '):].strip('\n')
            arr_target.remove(strtarget)
        elif 'all distortions: ' in line:
            arrdistort_all = line[len('all distortions: '):].strip('\n').split(';')
        elif 'Image norm: ' in line:
            strimagenorm = line[len('Image norm: '):].strip('\n')
        elif 'Classification:' in line:
            if 'original ' in line:
                strline_orig = line.strip('\n')
            else:
                strline_adv = line.strip('\n')
                stradv, correct2target, correct2other = get_line_stats(strtarget,strline_orig,strline_adv)
                strdistortion = arrdistort_all[int(stradv)]
        elif 'vote: ' in line:
            if 'original ' in line:
                itmp = line.find('original vote: ')+len('original vote: ')
                strvote_orig = line[itmp:].strip('\n')
            else:
                itmp = line.find('adv vote: ')+len('adv vote: ')
                strvote_adv = line[itmp:].strip('\n')
                strout = strtarget+','+stradv+','+strdistortion+','+strimagenorm+','+str(correct2target)+','+str(correct2other)+','+strvote_orig+','+strvote_adv
                print(strout)
                arr_out.append(strout)
    if (strFileout != ''):
        fOut = open(strFileout, 'w')
        fOut.write(strheader+'\n')
        strinput = arr_target[0]
        for strout in arr_out:
            fOut.write(strinput+','+strout+'\n')
        fOut.close()
        

def get_line_stats(strtarget,strline_orig,strline_adv):
    arrsplit_orig = strline_orig.split(',')
    arrsplit_adv = strline_adv.split(',')
    iadv = arrsplit_orig[0].find('target'+strtarget)+len('target'+strtarget)
    stradv = arrsplit_orig[0][iadv+4:]
    itmp = arrsplit_orig[1].find(': ')
    arrsplit_orig[1] = arrsplit_orig[1][itmp+2:]
    iadv = arrsplit_adv[0].find('target'+strtarget)+len('target'+strtarget)
    itmp = arrsplit_adv[1].find(': ')
    arrsplit_adv[1] = arrsplit_adv[1][itmp+2:]
    count2target = 0
    count2other = 0 
    for i in range(1,len(arrsplit_orig)):
        if arrsplit_adv[i] != arrsplit_orig[i]:
            if arrsplit_adv[i] == strtarget:
                count2target = count2target+1
            else:
                count2other = count2other+1
    return stradv, count2target, count2other


