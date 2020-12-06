"""
========================================================
Optimize the hyperparameters of code2vec
========================================================
"""

print("GO")
import numpy as np
print("NUMPY imported")
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
print("CONFIG imports done")
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
print("SMAC imports done")
# Import Code2vec model and config class
from tensorflow_model import Code2VecModel
from config import Config
print("IMPORTS DONE")
import os, shutil
def cleanup(cfg):
    folders = [cfg.MODEL_SAVE_PATH]
    folders = [("/".join(x.split('/')[:-1]) + '/') for x in folders]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def vec_from_cfg(cfg):
    """ Creates a C2V instance based on a configuration and evaluates it on the
    java-small dataset.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    """

    # We convert the cfg to dict, then translate boolean and power of 2 values:

    cfg = {k:cfg[k] for k in cfg if cfg[k]}
    cfg["SEPARATE_OOV_AND_PAD"] = True if cfg["SEPARATE_OOV_AND_PAD"] == "true" else False
    cfg["DEFAULT_EMBEDDINGS_SIZE"] = 2**cfg["DEFAULT_EMBEDDINGS_SIZE"]
    print("CFG", cfg)

    config = Config(set_defaults=True, load_from_args=True, verify=True, hyper_params=cfg)
    cleanup(config)
    model = Code2VecModel(config)
    model.train()
    cleanup(config)
    return model.evaluate().subtoken_f1  # Maximize!




# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible hps of code2vec and add them as variabled to our cs
MAX_CONTEXTS = UniformIntegerHyperparameter("MAX_CONTEXT", 100, 300, default_value=200)
MAX_TOKEN_VOCAB_SIZE = UniformIntegerHyperparameter("MAX_TOKEN_VOCAB_SIZE", 1000000, 2000000, default_value=1301136)
MAX_TARGET_VOCAB_SIZE = UniformIntegerHyperparameter("MAX_TARGET_VOCAB_SIZE", 200000, 300000, default_value=261245)
MAX_PATH_VOCAB_SIZE = UniformIntegerHyperparameter("MAX_PATH_VOCAB_SIZE", 900000, 1000000, default_value=911417)
DEFAULT_EMBEDDINGS_SIZE = UniformIntegerHyperparameter("DEFAULT_EMBEDDINGS_SIZE", 5, 10 , default_value=7)
SEPARATE_OOV_AND_PAD = CategoricalHyperparameter("SEPARATE_OOV_AND_PAD", ["true", "false"], default_value="false")
DROPOUT_KEEP_RATE = UniformFloatHyperparameter("DROPOUT_KEEP_RATE", 0.001, 1.0, default_value=0.75)

cs.add_hyperparameter(MAX_CONTEXTS)
cs.add_hyperparameter(MAX_TOKEN_VOCAB_SIZE)
cs.add_hyperparameter(MAX_TARGET_VOCAB_SIZE)
cs.add_hyperparameter(MAX_PATH_VOCAB_SIZE)
cs.add_hyperparameter(DEFAULT_EMBEDDINGS_SIZE)
cs.add_hyperparameter(SEPARATE_OOV_AND_PAD)
cs.add_hyperparameter(DROPOUT_KEEP_RATE)

# There are some hyperparameters shared by all kernels

print("ConfigSpace has been setup")

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
    "runcount-limit": 50,  # max. number of function evaluations; for this example set to a low number
    "cs": cs,  # configuration space
    "deterministic": "true"
    })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = vec_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=vec_from_cfg)

incumbent = smac.optimize()

inc_value = vec_from_cfg(incumbent)

print("Optimized Value: %.2f" % (inc_value))

# We can also validate our results (though this makes a lot more sense with instances)
smac.validate(config_mode='inc',  # We can choose which configurations to evaluate
        # instance_mode='train+test',  # Defines what instances to validate
        repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
        n_jobs=1)  # How many cores to use in parallel for optimization
