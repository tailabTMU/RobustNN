from process_results import process_all_results, extract_stats, get_transfer_stats, get_stats_by_bucket_arrfiles


#process results for superimposition attacks------------------------------------------------------
#if the tests were run in small batches, the output summary txt files should be put a single folder
strTest = 'models_ensemble_mnist_T10-500/sup3_noisy_att/'
#process the files (generated from test_ensemble_noisylogit_superimposed_opt) in the above folder into a table, 
#will generate a master.txt file
process_all_results(strTest, bSupImp=True)

#extract stats for superimposition attacks
strMaster = strTest+'master.txt'
strTeacherPrefix = '500mnist_3000epoch_50_teachers_'
arrTeachers = [strTeacherPrefix+str(i) for i in range(50)]
#extract statistics from the master.txt file
extract_stats(strMaster, arrTeachers)
#------------------------------------------------------------------------------------------------


#process results for transferability test--------------------------------------------------------
stradvvote='adv_vote'
strorigvote='orig_vote'
str2target='correct_to_target'
str2other='correct_to_other'
strdistort='Distortion'
strnorm='Image_norm'
strinput='Input'
strtarget='Target'

strfolder1 = 'models_ensemble_mnist_T10-500/'
strfileprefix1 = 'test_img_all50model_'
arrfilenames1 = []
for i in range(1,10):
    tmp = str(i)+'-'+str(i)
    strfilename = strfolder1+strfileprefix1+tmp+'/testoutput_'+tmp+'_summary.txt'
    strFileout = strfolder1+strfileprefix1+tmp+'/summary_stats.txt'
    arrfilenames1.append(strFileout)
    #this creates a summary file containing statistics of the raw output file generated from test_ensemble_noisylogit_superimposed_all_byinput
    get_transfer_stats(strfilename,strFileout=strFileout)
#this groups the stats from the above summary files by different distortion bin sizes
get_stats_by_bucket_arrfiles(arrfilenames1, 20)
#------------------------------------------------------------------------------------------------
