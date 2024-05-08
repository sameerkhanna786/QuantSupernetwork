import copy
import logging
import functools

from torch.optim import SGD
from torch.optim import Adam
from torch.optim import ASGD
from torch.optim import Adamax
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import RMSprop
from adabound import AdaBound
from bitsandbytes.optim import *

logger = logging.getLogger('nas_seg')

key2opt =  {'sgd': SGD,
            'adam': Adam,
            'asgd': ASGD,
            'adamax': Adamax,
            'adadelta': Adadelta,
            'adagrad': Adagrad,
            'rmsprop': RMSprop,
            'adabound': AdaBound}

key2opt_low_prec = {
    'sgd': SGD8bit,
    'adam': Adam8bit,
    'adagrad': Adagrad8bit,
    'rmsprop': RMSprop8bit
}

def get_optimizer(cfg, phase='searching', optimizer_type='optimizer_model', low_prec_optim = False):

    if cfg[phase][optimizer_type] is None:
        logger.info("Using SGD optimizer")
        return SGD
    else:
        opt_name = cfg[phase][optimizer_type]['name']

        search_dict = key2opt
        if low_prec_optim:
            search_dict = key2opt_low_prec

        if opt_name not in search_dict:
            raise NotImplementedError('Optimizer {} not implemented'.format(opt_name))

        logger.info('Using {} optimizer'.format(opt_name))
        if low_prec_optim:
            logger.info('Using low precision version of optimizer')
        return search_dict[opt_name]
