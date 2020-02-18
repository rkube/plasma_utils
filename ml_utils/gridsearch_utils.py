#-*- Encoding: UTF-8 -*-

# Helper class for gridsearch

# Helper class for gridsearch

import time
import yaml
import hashids

class gridsearch_params(dict):
    def __init__(self, run_id, model_name, model_dims, nn_idx_list,
                 loss_fun, optim, lr_start, batch_size, epochs):
        hid = hashids.Hashids(min_length=10)
        
        # Get a unique run id
        
        self._hparams = {"run_id": hid.encode(int(time.time())),
                         "model_name": model_name,
                         "model_dims": model_dims,
                         "nn_idx_list": nn_idx_list,
                         "loss_fun": loss_fun,
                         "optim": optim,
                         "lr_start": lr_start,
                         "batch_size": batch_size,
                         "epochs": epochs}
       
        fname = f"logs/ml_data_case_2/{self._hparams['run_id']}.yaml"
        print(fname)
        with open(fname, "w") as stream:
            yaml.dump(self._hparams, stream)
            stream.close()
    
    def __str__(self):
        return f"I am a gridsearch_param run_id {self._hparams['run_id']}"
    
    
    def logfile_name(self):
        return f"logs/ml_data_case_2/{self._hparams['run_id']}.log"
    
        
    def plot_basename(self):
        return f"plots/ml_data_case_2/"
    
    
    def __setitem__(self, key, item):
        self._hparams[key] = item

    def __getitem__(self, key):
        return self._hparams[key]

    def __repr__(self):
        return repr(self._hparams)

    def __len__(self):
        return len(self._hparams)



# End of file gridsearch_utils.py