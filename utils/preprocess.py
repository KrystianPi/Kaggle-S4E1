from sklearn.impute import SimpleImputer
from optuna import Trial

def instantiate_numerical_simple_imputer(trial : Trial, fill_value : int=-1) -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'numerical_strategy', ['mean', 'median', 'most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)

def instantiate_categorical_simple_imputer(trial : Trial, fill_value : str='missing') -> SimpleImputer:
  strategy = trial.suggest_categorical(
    'categorical_strategy', ['most_frequent', 'constant']
  )
  return SimpleImputer(strategy=strategy, fill_value=fill_value)