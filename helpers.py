import os
import numpy as np
import scipy


class EEGDataloader:
    def __init__(self, path):
        # path to root data directory
        self.path = path
        
        # get absolute file path for each recorded subject data
        self.file_paths = {} # subject_index:subject_data_filepath
        self._get_subject_data_paths()

        self.eeg = None
        self.label = None
        
    def _get_subject_data_paths(self):
        for i in range(1,51):
            dir_name = 'sub-'
            if len(str(i)) < 2:
                dir_name = dir_name + '0' + str(i)
            else:
                dir_name = dir_name + str(i)
        
            dir_path = os.path.join(self.path,dir_name)
            for file in os.listdir(dir_path):
                self.file_paths[i] = os.path.join(dir_path,file)
                
    def get_subject_data(self, subj_ind):
        # iterate over each subject and load data
        mat_dict = scipy.io.loadmat(self.file_paths[subj_ind])
        eeg0 = mat_dict['eeg'][0][0][0]
        eeg0 = np.transpose(eeg0,(0,2,1)) # shape: (40,4000,33) => (number of trials, number of samples, number of channels)
        
        data = eeg0.reshape(-1, eeg0.shape[2])
        # print(np.shares_memory(data, eeg0))
        self.label = mat_dict['eeg'][0][0][1] # shape: (40,1)
        
        channels = [*np.arange(0,17), *np.arange(18,30)]
        
        # grab the indexes connected to the event triggers
        trigger_indxs = np.where(data[:, 32] == 2)[0] # (trial index, time index)
        
        self.eeg = np.zeros((2500,29,40))
        
        for i in range(0,len(trigger_indxs)):
            beg_ind = trigger_indxs[i] - 500
            end_ind = beg_ind + 2500
            # get samples that are around the event trigger
            self.eeg[:,:,i] = data[beg_ind:end_ind, channels]

        return self.eeg, self.label

    def get_trials_by_label(self,label_selection):
        """
        Seletions:
        ----------
        1 => left hands
        2 => right hands
        """
        # get indexes based on label selection
        data_indxs = np.where(self.label == label_selection)[0] 
        
        # slice for trials based on left or right hands
        data = self.eeg[:,:,data_indxs]

        return data