import copy
from pathlib import Path
import os
import numpy as np
from numpy.lib.format import open_memmap
import sys

import spikeextractors as se
from ..basesorter import BaseSorter
from ..utils.shellscript import ShellScript
from ..sorter_tools import recover_recording

try:
    import yass
    HAVE_YASS = True
except ImportError:
    HAVE_YASS = False


class YassSorter(BaseSorter):
    """
    """

    sorter_name = 'yass'
    requires_locations = False

    _default_params = {
        }

    _params_description = {
    }

    sorter_description = """Yass description; link to biorxiv"""

    installation_mesg = """\nTo use Yass run:\n
        >>> pip install yass-algorithm

        More information ...
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)
    
    @classmethod
    def is_installed(cls):
        return YASS
    
    @staticmethod
    def get_sorter_version():
        return yass.__version__

   def _setup_recording(self, recording, output_folder):
        p = self.params
        source_dir = Path(output_folder).parent

        #################################################################
        #################### UPDATE ROOT FOLDER #########################
        #################################################################
        # float(self._recording.sample_rate.rescale('Hz').magnitude)
        self.params['data']['root_folder'] = output_folder
        #self.params['data']['geometry'] = 'geom.csv'
        
        #################################################################
        #################### GEOMETRY FILE GENERATION ###################
        #################################################################
        # save prb file
        # note: only one group here, the split is done in basesorter
        probe_file_csv = os.path.join(output_folder,'geom.csv')
        probe_file_txt = os.path.join(output_folder,'geom.txt')
        # ALESSIO .saveto probe saved .prb file; have to d thisourselves.
        #  
        adjacency_radius = -1
        recording.save_to_probe_file(probe_file_csv, 
                                     grouping_property=None,
                                     radius=adjacency_radius)
        
        import csv

        with open(probe_file_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            geom_txt = np.float32(np.vstack(csv_reader))
            np.savetxt(probe_file_txt, geom_txt)
        
        #################################################################
        #################### UPDATE SAMPLING RATE #######################
        #################################################################
        # float(self._recording.sample_rate.rescale('Hz').magnitude)
        self.params['recordings']['sampling_rate'] = recording.get_sampling_frequency()
        
        
        #################################################################
        #################### UPDATE N_CHAN  #############################
        #################################################################
        self.params['recordings']['n_channels'] = recording.get_num_channels()
        
        
        #################################################################
        #################### SAVE RAW INT16 data ########################
        #################################################################
        # ALESSIO Look at Kilosort 
        # There is alrady an extractor se.Mea1kRecordingExtractor()
        # all the functions are there already to concatenate in time;
        # multi-recording time extractor;
        
        # save binary file; THIS IS FROM KILOSORT
        input_file_path = os.path.join(output_folder, 'data.bin')
        
        recording.write_to_binary_dat_format(input_file_path, 
                                             dtype='int16', 
                                             chunk_mb=500) # time_axis=0,1 for C/F order
        
        
        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                   'config.yaml')
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.params, file)
        
        #self.fname_config = fname_config
        
        # ALESSIO:
        # Expose more config file parameters that are sensiitive:
        #  e.g. spike width; smallest cluster; min firing rates;
        #  

    def _run(self, recording, output_folder):
        '''
        '''
        recording = recover_recording(recording)  # allows this to run on multiple jobs (not just multi-core)
        
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        yass sort {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))
        else:
            shell_cmd = '''
                        #!/bin/bash
                        yass sort {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))

        shell_script = ShellScript(shell_cmd, 
                                   script_path=os.path.join(output_folder,self.sorter_name),
                                   log_path=os.path.join(output_folder,self.sorter_name+'.log'), 
                                   verbose=self.verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('yass returned a non-zero exit code')

    # Alessio might not want to put here; 
    # better option to have a parameter "tune_nn" which 
    def train(self, recording, output_folder):
        '''
        '''
        recording = recover_recording(recording)  # allows this to run on multiple jobs (not just multi-core)
        
        if 'win' in sys.platform and sys.platform != 'darwin':
            shell_cmd = '''
                        yass train {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))
        else:
            shell_cmd = '''
                        #!/bin/bash
                        yass train {config}
                    '''.format(config=os.path.join(output_folder,'config.yaml'))

        shell_script = ShellScript(shell_cmd, 
                                   script_path=os.path.join(output_folder,self.sorter_name),
                                   log_path=os.path.join(output_folder,self.sorter_name+'.log'), 
                                   verbose=self.verbose)
        shell_script.start()

        retcode = shell_script.wait()

        if retcode != 0:
            raise Exception('yass returned a non-zero exit code')  
            
    def NNs_update(self):
        ''' Update NNs to newly trained ones
        '''
        
        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.params['neuralnetwork']['denoise']['filename']= os.path.join(
                                            output_folder, 
                                            'tmp',
                                            'nn_train',
                                            'denoise.pt')
        self.params['neuralnetwork']['detect']['filename']= os.path.join(
                                            output_folder, 
                                            'tmp',
                                            'nn_train',
                                            'detect.pt')
        
        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                   'config.yaml')
        
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.params, file)
        
        
    def NNs_default(self):
        ''' Revert to default NNs
        '''
        #################################################################
        #################### UPDATE CONFIG FILE TO NEW NNS ##############
        #################################################################
        self.params['neuralnetwork']['denoise']['filename']= 'denoise.pt'
        self.params['neuralnetwork']['detect']['filename']= 'detect.pt'
        
        #################################################################
        #################### SAVE UPDATED CONFIG FILE ###################
        #################################################################
        fname_config = os.path.join(output_folder,
                                   'config.yaml')
        
        with open(fname_config, 'w') as file:
            documents = yaml.dump(self.params, file)   
            
    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.SpykingCircusSortingExtractor(folder_path=Path(output_folder) / 'recording')
        return sorting
