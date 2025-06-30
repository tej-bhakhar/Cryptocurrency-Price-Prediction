from .random_forest import RandomForest
from .arima import MyARIMA
from .LSTM import MyLSTM
from .xgboost import MyXGboost
from .orbit import Orbit
from .sarimax import Sarimax


MODELS = {'random_forest': RandomForest,
          'lstm': MyLSTM,
          'arima': MyARIMA,
          'orbit': Orbit,
          'sarimax': Sarimax,
          'xgboost': MyXGboost
          }
