#-*- Encoding: UTF-8 -*-

# Helper class for gridsearch

"""
See notes in OneNote:
CGConv Parameterscan
Nnconv2 Hyperparameterscan
"""

import time
import yaml
import hashids

from os.path import isdir, join

class gridsearch_params(dict):
    """A dictionary with some default keys and a method to store as a yaml file"""
    def __init__(self, basedir="/home/rkube/Projects/learning_xgc/logs/ml_data_case_2",
                 model_name=None,
                 model_dims=None,
                 loss_fun=None,
                 optim=None,
                 lr_start=None,
                 batch_size=None,
                 epochs=None):

        assert(isdir(basedir))
        hid = hashids.Hashids(min_length=10)

        self.basedir = basedir

        if model_name is None:
            print("gridsearch_params: model_name not set")
        if model_dims is None:
            print("gridsearch_params: model_name not set")
        if loss_fun is None:
            print("gridsearch_params: loss_fun not set")
        if optim is None:
            print("gridsearch_params: optim not set")
        if lr_start is None:
            print("gridsearch_params: lr_start not set")
        if batch_size is None:
            print("gridsearch_params: lr_start not set")
        if epochs is None:
            print("gridsearch_params: lr_start not set")

        # Get a unique run id
        self._hparams = {"run_id": hid.encode(int(time.time())),
                         "model_name": model_name,
                         "model_dims": model_dims,
                         "loss_fun": loss_fun,
                         "optim": optim,
                         "lr_start": lr_start,
                         "batch_size": batch_size,
                         "epochs": epochs}


    def __str__(self):
        return f"I am a gridsearch_param run_id {self._hparams['run_id']}"


    def write_logfile(self):
        fname = join(self.basedir, f"{self._hparams['run_id']}.yaml")
        with open(fname, "w") as stream:
            yaml.dump(self._hparams, stream)
            stream.close()

    def logfile_name(self):
        return join(self.basedir, f"{self._hparams['run_id']}.yaml")


    def __setitem__(self, key, item):
        self._hparams[key] = item


    def __getitem__(self, key):
        return self._hparams[key]


    def __repr__(self):
        return repr(self._hparams)


    def __len__(self):
        return len(self._hparams)

# End of file gridsearch_utils.py
