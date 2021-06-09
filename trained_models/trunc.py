# truncate vines for github (100 Mb limit)
# 45 trees >> 17 intrinsic dimensions, so the difference between these models and the full model is negligible
# this model is sufficient to reproduce all of the results in Figure 5G-H
import pickle as pkl

for old_name,new_name in zip(['Dataset','DatasetS'],['pYgX','pY']):
    with open(f'ST260_Day1_{old_name}_trained.pkl',"rb") as f:
        trained = pkl.load(f)
    trained['models'] = trained['models'][:20]
    trained['waics'] = trained['waics'][:20]
    with open(f'{new_name}_trained.pkl',"wb") as f:
        pkl.dump(trained,f)
