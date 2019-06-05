#from graphlib.keras_model import CarRatingPrediction
from graphlib.cryptographer import Cryptographer
from graphlib.volvo_stats import VolvoStats
from graphlib.types import Types
#from graphlib.carbinarizer import CarBinarizer
from graphlib.find_distribution import best_fit_distribution
import pandas as pd
import pandas_profiling as pp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import keras
from keras import optimizers
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from keras.layers import Embedding, Dense, Input, concatenate, dot, CuDNNLSTM, RepeatVector, Dropout, LeakyReLU, Flatten, BatchNormalization, Activation
from IPython.display import Image
from datetime import datetime
import tensorflow as tf
from tensorflow import set_random_seed
import numpy as np
import seaborn as sns
from scipy.stats import norm
from graphlib.cryptographer import Cryptographer
from graphlib.volvo_stats import VolvoStats
import pickle
set_random_seed(15)
import pandas as pd
import random
tf.__version__
import tensorflow as tf
import numpy as np
import cvxpy as cp
import cvxopt
import re
from functools import reduce
import pickle
print('exjobb data prepare loading...   ')
### USER COLUMNS
user_columns = ['antal_inpendlare', 'antal_utpendlare', 'förvärvs-arbetande',
       'ej_förvärvs-arbetande', 'äganderätt/småhus', 'bostadsrätt',
       'hyresrätt', 'övriga_inkl._uppgift_saknas', 'förgymnasial', 'gymnasial',
       'eftergymnasial_mindre_än_3_år',
       'eftergymnasial_3_år_eller_längre_inkl._forskarutbildning', '0-6_år',
       '7-15_år', '16-19_år', '20-24_år', '25-44_år', '45-64_år', '65-w_år',
       'låg_inkomst', 'medellåg_inkomst', 'medelhög_inkomst', 'hög_inkomst',
       'medianinkomst', 'sammanboende_med_barn', 'sammanboende_utan_barn',
       'ensamstående_med_barn', 'ensamstående_utan_barn', 'övriga_hushåll',
       'låg_köpkraft', 'medellåg_köpkraft', 'medelhög_köpkraft',
       'hög_köpkraft', 'median_köpkraft',
       'jordbruk,_skogsbruk,_jakt_och_fiske', 'tillverkning_och_utvinning',
       'energi_och_miljöverksamhet', 'byggverksamhet', 'handel',
       'transport_och_magasinering', 'hotell-_och_restaurangverksamhet',
       'information_och_kommunikation', 'finans-_och_försäkringsverksamhet',
       'fastighetsverksamhet', 'företagstjänster',
       'offentlig_förvaltning_och_försvar', 'utbildning',
       'vård_och_omsorg,_sociala_tjänster',
       'kulturella_och_personliga_tjänster_m.m.', 'okänd_verksamhet', '0_barn',
       '1_barn', '2_barn', '3+_barn']

vehicle_columns = ['TYPECODE', 'SALESVERSIONCODE', 'GEARBOXCODE', 'COLOUR', 'UPHOLSTERY', 'ENGINECODE','OPT_CODES']


#tf.enable_eager_execution()


from sklearn.preprocessing import MultiLabelBinarizer,OneHotEncoder,LabelBinarizer
import tqdm
class CarBinarizer(OneHotEncoder):
    def fit(self,values):
        """
            expects an array of tuples, which describes the different parts of a vehicle: 
            [('CAR_TYPE',235),('CAR_TYPE',225),('ENGINE',72)]
        """
        #print(f'needs fitting {values}')
        
        self.types = {}
        self.types_binarizer = {}
              
        self.inverse_lookup = {str(v):k for k,v in values}
        for k,v in values:
            if not k in self.types:
                self.types[k] = []
                if k == 'OPT':
                    self.types_binarizer[k] = MultiLabelBinarizer()
                else:
                    self.types_binarizer[k] = LabelBinarizer()

            if str(v) in self.types[k]:
                pass
            else:
                #print(f'{v} NOT IN {k}')
                self.types[k] += [str(v)]

        #print('TYPES',self.types)
        #print('INV L.',self.inverse_lookup)
        
        for k,values in self.types.items():
            #print(f'FITTING {values} to {k}, {type(values)}')
            if k == 'OPT':
                self.types_binarizer[k].fit([values])
            else:
                self.types_binarizer[k].fit(values)
        
    def _arr_to_dict(self,arr):
        #print(f'arr to dict {arr}')
        result = {}
        for k,v in arr:
            if isinstance(v,list):
                result[k] = [str(x) for x in v]
            else:
                result[k] = str(v)
       
        #print(f'arr to dict {result}')
                
        return result
        #return {k:str(v) if not isinstance(v,list) else k:v for k,v in arr}
 
    def transform(self,values):
        """
            Expect array of arrays, with tuples, such as : 
            [[('CAR_TYPE','235'),('ENGINE',72)],[('CAR_TYPE','235'),('ENGINE',72)]]
        """
       # print(f'needs transofmration {values}')
        
        transformed = []
        for arr in tqdm.tqdm(values):
            t=[]
            types_dict = self._arr_to_dict(arr)
            
            for key in self.types.keys():
                #print(key)
                assert key in self.types_binarizer
                assert key in types_dict
                val =types_dict[key]
                if isinstance(val,list):
                    #print(f'list value: {val} {type(val)}')
                    for x in val:
                        assert x in self.inverse_lookup
                else:
                    #print(f'singular value: {val} {type(val)}')
                    assert val in self.inverse_lookup
                #print(f'transforming {key} with {[val]}')
                
                if str(val) == 'nan':
                    val = [val]
                if isinstance(val, list):
                    #print(f'transform {val}')
                    t += [(key,self.types_binarizer[key].transform([val]))]
                else:
                    t += [(key,self.types_binarizer[key].transform([val]))]
                    
                #print(f'result {self.types_binarizer[key].transform([val])} ')
            transformed += [dict(t)]
        
        #print(f'transformed {transformed}')
        return np.array(transformed)
            
    
    def inverse_transform(self,arr_transformed):
        """
            inverse transforms a multihot encoded array of arrays, must have been generated from this instance, 
            otherwise the results is (probably) not valid
        """
        #print(f'invert_transfomr, {arr_transformed}')
        inverse_transformed = []
        for d in arr_transformed:
            #print(f'dict in invert {d}')
            assert len(d) == len(self.types)
            inv = []
            for k,v in d.items():
                inv += [(k,self.inverse_transform_type(v))]
            inverse_transformed += [inv]
        return inverse_transformed
    
    def inverse_transform_type(self, typ, transformed):
        return self.types_binarizer[typ].inverse_transform(transformed)
    
    def fit_transform(self,values):
        """
            expects the values to be a list of typed values, just like the transform method
            ex             [[('CAR_TYPE','235'),('ENGINE',72)]]
        """
        #print(f'needs fit+transform {values}')
        
        to_fit = set([])
        for value in values:
            for arr in value:
                if isinstance(arr[1],list):
                    [to_fit.add((arr[0], v)) for v in arr[1]]
                else:
                    to_fit.add(arr)
        self.fit(to_fit)
        return self.transform(values)


class VolvoStats:
    def __init__(self, df, debug=False, group_by='default'):
        self.debug = debug
        assert df is not None, "You must provide a dataframe if you don't want to reload it"
        self.volvo_df = df

        if group_by == 'default':
            self.group_by = self.volvo_df.MODEL_YEAR
        else: 
            self.group_by = group_by(df)

        self.agg_stats = {
            'counts': VolvoStats.top_k_values,
            'percentage': VolvoStats.top_k_count_percentage
        }

        self.mappings = {
            'ENGINECODE': Types.ENGINE,
            'TYPECODE': Types.CAR_TYPE,
            'SALESVERSIONCODE': Types.SALES_VERSION,
            'GEARBOXCODE': Types.GEARBOX,
            'COLOUR': Types.COL,
            'UPHOLSTERY': Types.UPH,
            'OPT_CODES': Types.OPT,
        }

        for key in self.mappings.keys():
            assert key in self.volvo_df.columns, "The dataframe provided does not contain column {}".format(
                key)
        self.cache = {}

        self.agg = {}
        for key in self.mappings:
            if key is 'OPT_CODES':
                self.agg[key] = {
                    'counts':VolvoStats.tok_k_values_list,
                    'percentage':VolvoStats.top_k_percentage_list
                }
            else:
                self.agg[key] = self.agg_stats

        self._init_statistics()

    def debug_print(self, *msg):
        if self.debug:
            print(msg)

    def _init_statistics(self):
        self.volvo_agg_df = self.volvo_df.groupby(self.group_by).agg(self.agg)
        for year in self.volvo_agg_df.index:
            for code, typ in self.mappings.items():
                stats = self.volvo_agg_df.loc[year][code]
                if not year in self.cache:
                    self.cache[year] = {}
                self.cache[year][typ] = dict(
                    zip(stats['counts'], stats['percentage']))

    @staticmethod
    def top_k_count(series):
        return ",".join(map(lambda x: str(x).strip(), series.value_counts())).replace(',,', ',')

    @staticmethod
    def top_k_values(series):
        return list(map(lambda x: x, series.value_counts().index))

    @staticmethod
    def top_k_count_percentage(series):
        total = series.value_counts().sum()
        values = [(v/total)
                  for _, v in series.value_counts().to_dict().items()]
        return values
    
    def top_k_list(series_w_list):
        series_dict= series_w_list.to_dict()
        items = []
        for list_items in series_dict.values():
            #print('kv',list_items)
            if isinstance(list_items,list):
                items += list_items                
        return Counter(items)

    def tok_k_values_list(series):
        return list(map(lambda x: x[0], top_k(series).most_common()))

    def top_k_percentage_list(series):
        series_length=len(series.to_dict().keys())
        return list(map(lambda x: x[1]/series_length, top_k(series).most_common()))

    @property
    def stats(self):
        return self.cache

def np_sed_loss(y_true, y_pred):
        return ((y_true-y_pred)**2).mean()

from collections import deque
class Evaluator:
    def __init__(self,test_df,iterations,sample_size=1000):
        self.test_df = test_df
        self.evaluation_results = {}
        self.iterations = iterations
        self.sample_size=sample_size
        self.queue = deque()
        
    def run(self, predictor, name_of_run,iterations = None):
        run_itr = iterations if not iterations is None else self.iterations
        print(f'Running evaluation on {name_of_run} over {run_itr} iterations')
        runs = 1
        self.evaluation_results[name_of_run] = []
        for run in tqdm_notebook(range(run_itr)):
            try: 
                print(f'Running test n.{run+1}')
                subset = self.test_df.copy().sample(self.sample_size,replace=True)
                predictions = predictor.predict(subset[user_columns])
                subset['predicted_comp_vec'] = pd.Series(predictions,index=subset.index)
                subset['predicted_comp_vec_concat'] = subset['predicted_comp_vec'].apply(lambda x: np.concatenate(x,axis=1))
                subset['actual_comp_vec_concat'] = subset[predictor.target_columns].apply(lambda x: np.concatenate(x,axis=1),axis=1)
                subset['row_loss'] = subset[['predicted_comp_vec_concat','actual_comp_vec_concat']].apply(lambda x: np_sed_loss(x['actual_comp_vec_concat'],x['predicted_comp_vec_concat']) ,axis=1)
                self.evaluation_results[name_of_run] += [subset.row_loss.values.mean()]
            except Exception as e:
                print(f'Error processing in run {run+1}...',e)    
        #self.evaluation_results[name_of_run] = np.average(self.evaluation_results[name_of_run])
        return self.evaluation_results[name_of_run]
    
    def enqueue(self,resolver,name_of_run,iterations=None):
        self.queue.append((resolver,name_of_run,iterations))
        
    def run_queue(self):
        from multiprocessing import Pool
        for predictor,name,iterations in tqdm_notebook(self.queue):
            self.run(predictor,name,iterations)
        self.queue.clear()
        return self.history
    
    @property
    def history(self):
        return self.evaluation_results.copy()

PNO12='pno12'
PNO34='pno34'
CIS='cis'
class Pno34ModelResolver:
    def __init__(self,automodel,carbinarizer,pno12_solver,threshold=0,pnotype='pno34',use_solver=False):
        self.pno12_solver = pno12_solver
        self.automodel = automodel
        self.carbinarizer = carbinarizer
        self.threshold = threshold
        self.pnotype = pnotype
        self.use_solver = use_solver
    
    def get_onehot(self,typ, predictions):
        oh = np.zeros(len(self.carbinarizer.types_binarizer[typ].classes_))
        oh[predictions.argmax()] = 1
        oh = oh.reshape(1,-1)
        return self.carbinarizer.types_binarizer[typ].inverse_transform(oh)
    
    def get_all_with_weights(self,typ, predictions):
        eye = np.eye(predictions.shape[0])
        #print(eye)
        for idx in range(len(predictions)):
            yield (self.carbinarizer.types_binarizer[typ].inverse_transform(eye[idx].reshape(1,-1))[0],predictions[idx])
    def get_all_with_weights_opt(self,typ,predictions):
        ones = np.ones(predictions.shape).reshape(1,-1)
        inv_trans = self.carbinarizer.types_binarizer[typ].inverse_transform(ones)[0]
        for idx in range(len(inv_trans)):
            yield (inv_trans[idx], 1 if predictions[idx] > self.threshold else 0)

            
    def _predict_pno12(self,ct,eng,sv,gb):
        real_ct = self.get_onehot('CAR_TYPE',ct)
        real_sv = self.get_onehot('SALES_VERSION',sv)
        real_eng = self.get_onehot('ENGINE',eng)
        real_gb = self.get_onehot('GEARBOX',gb)
        ct_oh = self.carbinarizer.types_binarizer['CAR_TYPE'].transform(real_ct)
        sv_oh = self.carbinarizer.types_binarizer['SALES_VERSION'].transform(real_sv)
        eng_oh = self.carbinarizer.types_binarizer['ENGINE'].transform(real_eng)
        gb_oh = self.carbinarizer.types_binarizer['GEARBOX'].transform(real_gb)
        return (ct_oh, sv_oh, eng_oh, gb_oh)
    
    def _predict_pno34(self,ct,eng,sv,gb,col,uph):
        real_ct = self.get_onehot('CAR_TYPE',ct)
        real_sv = self.get_onehot('SALES_VERSION',sv)
        real_eng = self.get_onehot('ENGINE',eng)
        real_gb = self.get_onehot('GEARBOX',gb)

        
        if self.use_solver:
            real_uphs = self.get_all_with_weights_opt('UPH',uph)
            real_cols = self.get_all_with_weights_opt('COL',col)
            #print(real_opts)
            opt, car, vec, res = self.pno12_solver.solve(real_ct,real_sv,real_eng,real_gb,real_cols,real_uphs,[])
            _,ct,eng,sv,gb = re.findall(r'(..PNO12..)(...)_(..)_(..)_(.)',car[0])[0]
            color = list(map(lambda y: y[-5:],filter(lambda x: 'COL' in x,car)))
            uph = list(map(lambda y: y[-6:],filter(lambda x: 'UPH' in x,car)))
            ct_oh = self.carbinarizer.types_binarizer['CAR_TYPE'].transform([ct])
            sv_oh = self.carbinarizer.types_binarizer['SALES_VERSION'].transform([eng])
            eng_oh = self.carbinarizer.types_binarizer['ENGINE'].transform([sv])
            gb_oh = self.carbinarizer.types_binarizer['GEARBOX'].transform([gb])
            color_oh = self.carbinarizer.types_binarizer['COL'].transform(color)
            uph_oh = self.carbinarizer.types_binarizer['UPH'].transform(uph)
            
            return (ct_oh, sv_oh, eng_oh, gb_oh,color_oh,uph_oh)
        
        real_uphs = self.get_onehot('UPH',uph)
        real_cols = self.get_onehot('COL',col)
        ct_oh = self.carbinarizer.types_binarizer['CAR_TYPE'].transform(real_ct)
        sv_oh = self.carbinarizer.types_binarizer['SALES_VERSION'].transform(real_sv)
        eng_oh = self.carbinarizer.types_binarizer['ENGINE'].transform(real_eng)
        gb_oh = self.carbinarizer.types_binarizer['GEARBOX'].transform(real_gb)
        uph_oh = self.carbinarizer.types_binarizer['UPH'].transform(real_uphs)
        color_oh = self.carbinarizer.types_binarizer['COL'].transform(real_cols)
        
        
        
        
        return (ct_oh, sv_oh, eng_oh, gb_oh,color_oh,uph_oh)
    
    def _predict_cis(self,ct,eng,sv,gb,col,uph,opt):
        real_ct = self.get_onehot('CAR_TYPE',ct)
        real_sv = self.get_onehot('SALES_VERSION',sv)
        real_eng = self.get_onehot('ENGINE',eng)
        real_gb = self.get_onehot('GEARBOX',gb)
        
        real_ct = self.get_onehot('CAR_TYPE',ct)
        real_sv = self.get_onehot('SALES_VERSION',sv)
        real_eng = self.get_onehot('ENGINE',eng)
        real_gb = self.get_onehot('GEARBOX',gb)
        real_uphs = self.get_all_with_weights_opt('UPH',uph)
        real_cols = self.get_all_with_weights_opt('COL',col)
        
        
        if self.use_solver:
            #print(real_opts)
            real_opts = self.get_all_with_weights_opt('OPT',opt)
            
            opt, car, vec, res = self.pno12_solver.solve(real_ct,real_sv,real_eng,real_gb,real_cols,real_uphs,real_opts)
            _,ct,eng,sv,gb = re.findall(r'(..PNO12..)(...)_(..)_(..)_(.)',car[0])[0]
            color = list(map(lambda y: y[-5:],filter(lambda x: 'COL' in x,car)))
            uph = list(map(lambda y: y[-6:],filter(lambda x: 'UPH' in x,car)))
            opts = list(map(lambda y: y[-6:],filter(lambda x: 'OPT' in x,car)))

            ct_oh = self.carbinarizer.types_binarizer['CAR_TYPE'].transform([ct])
            sv_oh = self.carbinarizer.types_binarizer['SALES_VERSION'].transform([eng])
            eng_oh = self.carbinarizer.types_binarizer['ENGINE'].transform([sv])
            gb_oh = self.carbinarizer.types_binarizer['GEARBOX'].transform([gb])
            color_oh = self.carbinarizer.types_binarizer['COL'].transform(color)
            uph_oh = self.carbinarizer.types_binarizer['UPH'].transform(uph)
            opts_oh = self.carbinarizer.types_binarizer['OPT'].transform([opts])

            return (ct_oh, sv_oh, eng_oh, gb_oh,color_oh,uph_oh,opts_oh)
        
        else:
            real_uphs = self.get_onehot('UPH',uph)
            real_cols = self.get_onehot('COL',col)
            real_opts = list(map(lambda x: x[0],filter(lambda y: y[1] == 1,self.get_all_with_weights_opt('OPT',opt))))
            ct_oh = self.carbinarizer.types_binarizer['CAR_TYPE'].transform(real_ct)
            sv_oh = self.carbinarizer.types_binarizer['SALES_VERSION'].transform(real_sv)
            eng_oh = self.carbinarizer.types_binarizer['ENGINE'].transform(real_eng)
            gb_oh = self.carbinarizer.types_binarizer['GEARBOX'].transform(real_gb)
            uph_oh = self.carbinarizer.types_binarizer['UPH'].transform(real_uphs)
            color_oh = self.carbinarizer.types_binarizer['COL'].transform(real_cols)
            opt_oh = self.carbinarizer.types_binarizer['OPT'].transform([real_opts])
            return (ct_oh, sv_oh, eng_oh, gb_oh,color_oh,uph_oh,opt_oh)

        
    def predict(self, context):
        #ctmodel,engmodel,svmodel,gbmodel,colmodel,uphmodel
        predictions = self.automodel.predict(context)
        ct_predictions = predictions[0]
        eng_predictions = predictions[1]
        sv_predictions = predictions[2]
        gb_predictions = predictions[3]
        col_predictions = predictions[4]
        uph_predictions = predictions[5]
        opt_predictions = predictions[6]
        all_pno12 = {}
        errors = 0
        error_tuples = []

        

        for ct,sv,eng,gb,col,uph,opt in tqdm_notebook(list(zip(ct_predictions,\
                                            sv_predictions,\
                                            eng_predictions,\
                                            gb_predictions,\
                                            col_predictions,\
                                            uph_predictions,\
                                            opt_predictions
                                                ))):
            try:
                if self.pnotype is PNO12:
                    yield self._predict_pno12(ct, eng, sv, gb)
                elif self.pnotype is PNO34:
                    yield self._predict_pno34(ct, eng, sv, gb,col,uph)
                elif self.pnotype is CIS:
                    yield self._predict_cis(ct,eng,sv,gb,col,uph,opt)
            except:
                error_tuples+=[(ct,sv,eng,gb,col,uph,opt)]
                errors += 1

        if errors != 0:
            print('could not process' ,errors,error_tuples)

    @property
    def target_columns(self):
        if self.pnotype is PNO12:
            return ['TYPECODE_onehot','SALESVERSIONCODE_onehot','ENGINECODE_onehot','GEARBOXCODE_onehot']
        elif self.pnotype is PNO34:
            return ['TYPECODE_onehot','SALESVERSIONCODE_onehot','ENGINECODE_onehot','GEARBOXCODE_onehot','COLOUR_onehot','UPHOLSTERY_onehot']
        elif self.pnotype is CIS:
            return ['TYPECODE_onehot','SALESVERSIONCODE_onehot','ENGINECODE_onehot','GEARBOXCODE_onehot','COLOUR_onehot','UPHOLSTERY_onehot','OPT_CODES_onehot']

class ComponentSolver:
    def __init__(self, rule_system):
        print('init...')
        try:
            i2c, c2i, A_ubs, b_ubs, A_eqs, b_eqs = rule_system
        except:
            raise Exception("'rule_system' must be a tuple consisting of 'i2c', 'c2i', 'A_ubs', 'b_ubs', 'A_eqs' and 'b_eqs'")

        self.i2c, self.c2i, self.A_ubs, self.b_ubs, self.A_eqs, self.b_eqs = i2c, c2i, A_ubs, b_ubs, A_eqs, b_eqs
        self.n = len(self.i2c)
    
    def solve(self, w, obj_cb=lambda w,x: w@x, solver=cp.GLPK_MI):

        print('solving...')
        # Define and solve the CVXPY problem.
        x = cp.Variable(n, boolean=True)
        prob = cp.Problem(cp.Minimize(cp.norm(ws[ct].T@x - w)), [
            A_ubs@x <= b_ubs, 
            A_eqs@x == b_eqs
        ])
        prob.solve(solver=cp.GUROBI)

        try:
            items = self.i2c[np.argwhere(x.value == 1).T[0].tolist()]
            res = (prob.status, items, x.value, prob.value)
        except:
            res = (prob.status, None, None)

        return res

class SolverData:
    def __init__(self):
        self.all_pno_components = {
            'cartypes':set([]),
            'salesversions':set([]),
            'engines':set([]),
            'gearboxes':set([]),
            'uph':set([]),
            'col':set([])
        }
        self.solver_data = pickle.load(open("/home/rocket/dev/jupyter/solver_data.pkl", "rb"))
        
        self.pno12s = []
        self.colors=[]
        self.uphs=[]
        self.opts=[]
        self.pno12s_ct={}
        for ct, values in self.solver_data['css'].items():
            self.pno12s+=list(filter(lambda x: 'PNO' in x[0],values.c2i.items()))
            self.colors+=list(filter(lambda x: 'COL' in x[0],values.c2i.items()))
            self.uphs+=list(filter(lambda x: 'UPH' in x[0],values.c2i.items()))
            self.opts+=list(filter(lambda x: 'OPT' in x[0],values.c2i.items()))


        self.uphs = list(map(lambda x: ('UPH',x[0][-6:]),self.uphs))
        self.colors = list(map(lambda x: ('COL',x[0][-5:]),self.colors))
        self.opts = list(map(lambda x: ('OPT',x[0][-6:]),self.opts))
        for pno12 in self.pno12s:
            _,ct,eng,sv,gb = re.findall(r'(..PNO12..)(...)_(..)_(..)_(.)',pno12[0])[0]
            self.all_pno_components['cartypes'].add(ct)
            self.all_pno_components['salesversions'].add(sv)
            self.all_pno_components['engines'].add(eng)
            self.all_pno_components['gearboxes'].add(gb)

        self.total_length=0
        self.total_length += reduce(lambda x,y:x+y,[len(x) for x in self.all_pno_components.values() ])
        self.total_length
        #print('pno12s', self.one_hot_encoder.classes_)
        #self.one_hot_encoder.fit()
        self._build_matrix()
        
    def _build_matrix(self):
        self.matrix = np.zeros((len(self.pno12s),self.total_length),dtype=np.int8)
        
        self.ct_index=list(self.all_pno_components['cartypes'])
        self.sv_index=list(self.all_pno_components['salesversions'])
        self.eng_index=list(self.all_pno_components['engines'])
        self.gb_index=list(self.all_pno_components['gearboxes'])

        for idx in range(len(self.pno12s)):

            _,ct,eng,sv,gb = re.findall(r'(..PNO12..)(...)_(..)_(..)_(.)',self.pno12s[idx][0])[0]
            ct_i = self.ct_index.index(ct)
            sv_i = len(self.ct_index)+self.sv_index.index(sv)
            eng_i = len(self.ct_index)+len(self.sv_index)+self.eng_index.index(eng)
            gb_i = len(self.ct_index)+len(self.sv_index)+len(self.eng_index)+self.gb_index.index(gb)
            self.matrix[idx,ct_i] = 1
            self.matrix[idx,sv_i] = 1
            self.matrix[idx,eng_i] = 1
            self.matrix[idx,gb_i] = 1
            
    def find_nearest_pno12(self,ct,sv,eng,gb):
        ct_i = self.ct_index.index(ct)
        sv_i = len(self.ct_index)+self.sv_index.index(sv)
        eng_i = len(self.ct_index)+len(self.sv_index)+self.eng_index.index(eng)
        gb_i = len(self.ct_index)+len(self.sv_index)+len(self.eng_index)+self.gb_index.index(gb)
        predicted=np.zeros(self.matrix.shape[1])
        predicted[ct_i] = 1
        predicted[sv_i] = 1
        predicted[eng_i] = 1
        predicted[gb_i] = 1
        row = np.sum(np.absolute(np.subtract(self.matrix,predicted)),axis=1).argmin()
        return self.pno12s[row]
    
    def filter_values_by_pno_type(self, ct,sv,eng,gb,typ,values):
        c2i = self.solver_data['css'][ct].c2i
        #print('ALL',list(filter(lambda x: typ in x[0], c2i.items())))
        theoretical_values = [(f'__{typ}__2019{ct}{eng}{sv}{gb}{v[0]}',v[1]) for v in values]
        #print(f'theoretircal {theoretical_values}')
        real_values = [val for val in theoretical_values if val[0] in map(lambda x: x[0], c2i.items())]
        return real_values

    def get_onehot_pno12_for_ct(self,ct,sv,eng,gb):
        pno12, idx = self.find_nearest_pno12(ct,sv,eng,gb)
        #search
        for ct, values in self.solver_data['css'].items():
            i2c  = values['i2c']
            if i2c[idx] == pno12:
                #match
                pno12s = list(filter(lambda x: 'PNO12' in x[0],values.c2i.items()))
                onehot = np.zeros(len(pno12s))
                onehot[idx] = 1
                return onehot
            

    def get_onehot_for_ct(self,ct,typ,value):
        c2i = self.solver_data['css'][ct].c2i
        
        typ_filtered = list(filter(lambda x: typ in x[0],c2i.items()))
        one_hot = np.zeros(len(typ_filtered))
        index_typ_mapping = {}
        
        for col, idx in typ_filtered:
            position = typ_filtered.index((col,idx))
            index_typ_mapping[position] = idx
           # if value 
    
    def invert_components_vec(self,ct,comps):
        pass

    def parse_pno12_str(self,pno12):
        _,ct,eng,sv,gb = re.findall(r'(..PNO12..)(...)_(..)_(..)_(.)',pno12)[0]
        return ct,eng,sv,gb

    def resolve_solved_items(self,items):
        ct,eng,sv,gb = self.parse_pno12_str(items[0])

        col = map(lambda x: x[0][-5:],self.filter_values_by_pno_type(ct,sv,eng,gb,'COL',map(lambda x: (x[-5:],'_'),items)))
        uph = map(lambda x: x[0][-6:],self.filter_values_by_pno_type(ct,sv,eng,gb,'UPH',map(lambda x: (x[-6:],'_'),items)))
        opt = map(lambda x: x[0][-6:],self.filter_values_by_pno_type(ct,sv,eng,gb,'OPT',map(lambda x: (x[-6:],'_'),items)))
    
        return (ct,eng,sv,gb,list(col)[0],list(uph)[0],list(opt))

    def build_components_vec(self,ct,sv,eng,gb, cols, uphs, opts,debug=False):
        """
            cols = [('71200',0.006),...]
            uhps = [('RA00000',0.231)]
            opts = [('079000',0.231)]
            
            
        """
        pno12 = self.find_nearest_pno12(ct,sv,eng,gb)
        #print('pno12:',pno12)
        _,ct,eng,sv,gb = re.findall(r'(..PNO12..)(...)_(..)_(..)_(.)',pno12[0])[0]
        #print(ct,eng,sv,gb)
        c2i = self.solver_data['css'][ct].c2i
        #print(c2i)
        carcomps = np.zeros(len(c2i))
        carcomps[pno12[1]] = 1
        cols = self.filter_values_by_pno_type(ct,sv,eng,gb,'COL',cols)
        uph = self.filter_values_by_pno_type(ct,sv,eng,gb,'UPH',uphs)
        opts = self.filter_values_by_pno_type(ct,sv,eng,gb,'OPT',opts)
        
        for item, w in cols+uph+opts:
            if debug:
                print('=>',item,c2i[item],w)
            carcomps[c2i[item]] = w

        return ct,carcomps
        
    def solve(self,ct,sv,eng,gb,cols,uphs,opts,debug=False):
        if debug:
            print(ct,sv,eng,gb,cols,uphs,opts)
        ct,w = self.build_components_vec(ct,sv,eng,gb,cols,uphs,opts,debug)
        #return self.solver_data['css'][ct].solve(w,lambda w,x: -(cp.sum(cp.abs(w-x))))
        return self.solver_data['css'][ct].solve(w,lambda w,x: w@x)
        
    def invert_onehot_pno12_for_ct(self,ct,one_hot):
        argmax = one_hot.argmax()
        return self.solver_data['css'][ct]['i2c'][argmax]

pno12_solver = SolverData()
train=None
test=None
dev=None
train_ct=None
train_eng=None
train_sv=None
train_gb=None
train_col=None
train_uph=None
def set_data(use_new=False):
    if use_new == False:
        print('using old data...')
        train_ct = pd.read_pickle('TYPECODE_train.pickle')
        train_sv = pd.read_pickle('SALESVERSIONCODE_train.pickle')
        train_gb = pd.read_pickle('GEARBOXCODE_train.pickle')
        train_eng = pd.read_pickle('ENGINECODE_train.pickle')
        train_uph = pd.read_pickle('UPHOLSTERY_train.pickle')
        train_col = pd.read_pickle('COLOUR_train.pickle')

        train = pd.read_pickle('train.pickle')
        test = pd.read_pickle('test.pickle')
        dev = pd.read_pickle('dev.pickle')
    else:
        print('using new data...')
        train_ct = pd.read_pickle('TYPECODE_2_train.pickle')
        train_sv = pd.read_pickle('SALESVERSIONCODE_2_train.pickle')
        train_gb = pd.read_pickle('GEARBOXCODE_2_train.pickle')
        train_eng = pd.read_pickle('ENGINECODE_2_train.pickle')
        train_uph = pd.read_pickle('UPHOLSTERY_2_train.pickle')
        train_col = pd.read_pickle('COLOUR_2_train.pickle')

        train = pd.read_pickle('train_2.pickle')
        test = pd.read_pickle('test_2.pickle')
        dev = pd.read_pickle('dev_2.pickle')
    return train,test,dev,train_ct,train_eng,train_sv,train_gb,train_uph,train_col

print('data loaded ....')