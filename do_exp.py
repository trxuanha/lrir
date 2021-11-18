import os
import numpy as np
from subprocess import call
import argparse
import sys


# Adapted from: https://github.com/clinicalml/cfrnet/tree/master/cfr

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file, 'r') as f:
        for l in f:
            l = l.strip()
            if len(l) > 0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs) > 0:
                    k, v = (vs[0], eval(vs[1]))
                    if not isinstance(v, list):
                        v = [v]
                    cfg[k] = v
    return cfg


def sample_config(configs, num):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts), 1)[0]
        cfg_sample[k] = opts[c]
    cfg_sample['config_num'] = num
    return cfg_sample


def cfg_string(cfg):
    ks = sorted(cfg.keys())
    # cfg_str = ','.join(['%s:%s' % (k, str(cfg[k])) for k in ks])
    cfg_str = ''
    for k in ks:
        if k == 'config_num':
            continue
        elif cfg_str != '':
            cfg_str = cfg_str + ', ' + ('%s:%s' % (k, str(cfg[k])))
        else:
            cfg_str = ('%s:%s' % (k, str(cfg[k])))
    return cfg_str.lower()


def is_used_cfg(cfg, used_cfg_file):
    cfg_str = cfg_string(cfg)
    used_cfgs = read_used_cfgs(used_cfg_file)
    return cfg_str in used_cfgs


def read_used_cfgs(used_cfg_file):
    used_cfgs = set()
    with open(used_cfg_file, 'r') as f:
        for l in f:
            used_cfgs.add(l.strip())

    return used_cfgs


def save_used_cfg(cfg, used_cfg_file):
    with open(used_cfg_file, 'a') as f:
        cfg_str = cfg_string(cfg)
        f.write('%s\n' % cfg_str)


def run(cfg_file, scriptName, fold_num=-1):

    configs = load_config(cfg_file)
    
    configs['fold_num'] = fold_num
    
    cfg = sample_config(configs, num=1)
    outdir = 'config'
    flags = ' '.join('--%s %s' % (k, str(v)) for k, v in cfg.items())
    
    call('python ' + scriptName + ' %s' % flags, shell=True)


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print ('Usage: python do_exp.py <config file>')
    else:
        print('sys.argv[1]')
        print(sys.argv)
        
        if(len(sys.argv) == 4):
            print('len(sys.argv) == 4')
            run(sys.argv[1], sys.argv[2], sys.argv[3])
        else: 
            print('len(sys.argv) < 4')
            run(sys.argv[1], sys.argv[2], '0')
            
            
# python do_exp.py config/turnover.txt  deepsurv.py
