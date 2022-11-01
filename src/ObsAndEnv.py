import moxing as mox
### Copy single dataset from obs to training image###
def ObsToEnv(obs_data_url, data_dir):
    try:     
        mox.file.copy_parallel(obs_data_url, data_dir)
        print("Successfully Download {} to {}".format(obs_data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
    return 
### Copy the output model to obs###
def EnvToObs(train_dir, obs_train_url):
    try:
        mox.file.copy_parallel(train_dir, obs_train_url)
        print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
    return      