import sys
import train
import traceback
import warnings
from torch.cuda import empty_cache

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

exp_pref = sys.argv[1]
layer = int(sys.argv[2])

print(f'Starting {exp_pref} layer {layer}')

train.train_next_layer(exp_pref, layer)
train.merge_results(exp_pref, layer)
NN = train.transform2next_layer(exp_pref,layer,'cpu')
print(f"NN = {NN}")