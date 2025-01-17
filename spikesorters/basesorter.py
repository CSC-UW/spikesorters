"""
Here a proposal for the futur Sorter with class approach.

The main idea is to decompose all intermediate steps to get more
flexibility:
  * setup the recording (traces, output folder, and so...)
  * set parameters
  * run the sorter (with futur possibility to make it in separate env/container)
  * get the result (SortingExtractor)

One benfit shoudl to compare the "run" time between sorter without
the setup and getting result.

One new idea usefull for tridesclous and maybe other sorter would
a way to adapt params with datasets.


"""

import time
import copy
from pathlib import Path
import datetime
import json
import traceback
import shutil
import warnings
from joblib import Parallel, delayed

import numpy as np

import spikeextractors as se
from spikeextractors.baseextractor import _check_json
from .sorter_tools import SpikeSortingError


class BaseSorter:
    sorter_name = ''  # convinience for reporting
    SortingExtractor_Class = None  # convinience to get the extractor
    requires_locations = False
    compatible_with_parallel = {'loky': True, 'multiprocessing': True, 'threading': True}
    _default_params = {}
    _params_description = {}
    sorter_description = ""
    installation_mesg = ""  # error message when not installed

    def __init__(self, recording=None, output_folder=None, verbose=False,
                 grouping_property=None, delete_output_folder=False):

        assert self.is_installed(), """The sorter {} is not installed.
        Please install it with:  \n{} """.format(self.sorter_name, self.installation_mesg)
        if self.requires_locations:
            if 'location' not in recording.get_shared_channel_property_names():
                raise RuntimeError("Channel locations are required for this spike sorter. "
                                   "Locations can be added to the RecordingExtractor by loading a probe file "
                                   "(.prb or .csv) or by setting them manually.")

        self.verbose = verbose
        self.grouping_property = grouping_property
        self.params = self.default_params()

        if output_folder is None:
            output_folder = self.sorter_name + '_output'
        output_folder = Path(output_folder).absolute()

        # if output_folder.is_dir():
        #     shutil.rmtree(str(output_folder))

        if grouping_property is None:
            # only one groups
            self.recording_list = [recording]
            self.output_folders = [output_folder]
            if 'group' in recording.get_shared_channel_property_names():
                groups = recording.get_channel_groups()
                if len(groups) != len(np.unique(groups)) > 1:
                    print("WARNING! The recording contains several group. In order to spike sort by 'group' use "
                          "grouping_property='group' as argument.")
        else:
            # several groups
            if grouping_property not in recording.get_shared_channel_property_names():
                raise RuntimeError(f"'{grouping_property}' is not one of the channel properties.")
            self.recording_list = recording.get_sub_extractors_by_property(grouping_property)
            n_group = len(self.recording_list)
            self.output_folders = [output_folder / str(i) for i in range(n_group)]

        # make dummy location if no location because some sorter need it
        for recording in self.recording_list:
            if 'location' not in recording.get_shared_channel_property_names():
                print('WARNING! No channel location given. Add dummy location.')
                channel_ids = recording.get_channel_ids()
                locations = np.array([[0, i] for i in range(len(channel_ids))])
                recording.set_channel_locations(locations)

        # make folders
        for output_folder in self.output_folders:
            output_folder.mkdir(parents=True, exist_ok=True)

        self.delete_folders = delete_output_folder

    @classmethod
    def default_params(cls):
        return copy.deepcopy(cls._default_params)

    @classmethod
    def params_description(cls):
        return copy.deepcopy(cls._params_description)
        
    def set_params(self, **params):
        bad_params = []
        for p in params.keys():
            if p not in self._default_params.keys():
                bad_params.append(p)
        if len(bad_params) > 0:
            raise AttributeError('Bad parameters: ' + str(bad_params))
        self.params.update(params)

        # dump parameters inside the folder with json
        self._dump_params()

    def _dump_params(self):
        for output_folder, recording in zip(self.output_folders, self.recording_list):
            with open(str(output_folder / 'spikeinterface_params.json'), 'w', encoding='utf8') as f:
                params = dict()
                params['sorter_params'] = self.params
                params['recording'] = recording.make_serialized_dict()
                json.dump(_check_json(params), f, indent=4)

    def run(self, raise_error=True, parallel=False, n_jobs=-1, joblib_backend='loky'):
        for i, recording in enumerate(self.recording_list):
            self._setup_recording(recording, self.output_folders[i])

        # dump again params because some sorter do a folder reset (tdc)
        self._dump_params()

        now = datetime.datetime.now()

        log = {
            'sorter_name': str(self.sorter_name),
            'sorter_version': str(self.get_sorter_version()),
            'datetime': now,
            'runtime_trace': []
        }

        t0 = time.perf_counter()

        if parallel:
            assert self.compatible_with_parallel[joblib_backend], f"{self.sorter_name} is not compatible with " \
                                                                  f"joblib {joblib_backend} backend"

        if parallel and len(self.recording_list) > 1:
            if not np.all([recording.check_if_dumpable() for recording in self.recording_list]):
                raise RuntimeError("RecordingExtractor objects are not dumpable and can't be processed in parallel. "
                                   "Use parallel=False")

        try:
            if not parallel:
                for i, recording in enumerate(self.recording_list):
                    self._run(recording, self.output_folders[i])
            else:
                Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                    delayed(self._run)(rec.dump_to_dict(), output_folder)
                    for (rec, output_folder) in zip(self.recording_list, self.output_folders))

            t1 = time.perf_counter()
            run_time = float(t1 - t0)

        except Exception as err:
            if raise_error:
                raise SpikeSortingError(f"Spike sorting failed: {err}. You can inspect the runtime trace in "
                                        f"the {self.sorter_name}.log of the output folder.'")
            else:
                run_time = None
                log['error'] = True
                log['error_trace'] = traceback.format_exc()

        log['run_time'] = run_time

        # dump log inside folders
        for i in range(len(self.output_folders)):
            output_folder = self.output_folders[i]
            runtime_trace_path = output_folder / f'{self.sorter_name}.log'
            runtime_trace = []
            if runtime_trace_path.is_file():
                with open(runtime_trace_path, 'r') as fp:
                    line = fp.readline()
                    while line:
                        runtime_trace.append(line.strip())
                        line = fp.readline()
            log['runtime_trace'] = runtime_trace
            with open(str(output_folder / 'spikeinterface_log.json'), 'w', encoding='utf8') as f:
                json.dump(_check_json(log), f, indent=4)

        if self.verbose:
            if run_time is None:
                print('Error running', self.sorter_name)
            else:
                print('{} run time {:0.2f}s'.format(self.sorter_name, t1 - t0))

        return run_time

    @staticmethod
    def get_sorter_version():
        # need be implemented in subclass
        raise NotImplementedError
    
    @classmethod
    def is_installed(cls):
        # need be implemented in subclass
        raise NotImplementedError

    def _setup_recording(self, recording, output_folder):
        # need be implemented in subclass
        # this setup ONE recording (or SubExtractor)
        # this must copy (or not) the trace in the appropirate format
        # this must take care of geometry file (ORB, CSV, ...)
        raise NotImplementedError

    def _run(self, recording, output_folder):
        # need be implemented in subclass
        # this run the sorter on ONE recording (or SubExtractor)
        # this must run or generate the command line to run the sorter for one recording
        raise NotImplementedError

    @staticmethod
    def get_result_from_folder(output_folder):
        raise NotImplementedError

    def get_result_list(self, raise_error=True):
        sorting_list = []
        for i, _ in enumerate(self.recording_list):
            try:
                sorting = self.get_result_from_folder(self.output_folders[i])
                sorting_list.append(sorting)
            except Exception as err:
                if raise_error:
                    raise SpikeSortingError(f"Failed to load sorting output {i}")
                else:
                    warnings.warn(f"Sorting output {i} could not be loaded")
        return sorting_list

    def get_result(self, raise_error=True):
        sorting_list = self.get_result_list(raise_error=raise_error)
        
        if len(sorting_list) == 1:
            sorting = sorting_list[0]
        elif len(sorting_list) > 1:
            for i, sorting in enumerate(sorting_list):
                property_name = self.recording_list[i].get_channel_property(self.recording_list[i].get_channel_ids()[0],
                                                                            self.grouping_property)
                if sorting is not None:
                    for unit in sorting.get_unit_ids():
                        sorting.set_unit_property(unit, self.grouping_property, property_name)

            # reassemble the sorting outputs
            sorting_list = [sort for sort in sorting_list if sort is not None]
            multi_sorting = se.MultiSortingExtractor(sortings=sorting_list)
            sorting = multi_sorting
        else:
            raise SpikeSortingError(f"None of the sorting outputs could be loaded")

        if self.delete_folders:
            for out in self.output_folders:
                if self.verbose:
                    print("Removing ", str(out))
                shutil.rmtree(str(out), ignore_errors=True)

        sorting.set_sampling_frequency(self.recording_list[0].get_sampling_frequency())
        sorting.copy_epochs(self.recording_list[0])
        sorting.copy_times(self.recording_list[0])

        return sorting
