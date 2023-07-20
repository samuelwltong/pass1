import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


def create_processing_pipeline(config):
    
    if config['preprocessor_pipeline']['sampling'] == "smote":
        sampler_parameters = dict(config['smote_parameters'])
        sampler = SMOTE(**sampler_parameters)

    elif config['preprocessor_pipeline']['sampling'] == "oversample":
        sampler_parameters = dict(config['oversample_parameters'])
        sampler = RandomOverSampler(**sampler_parameters)

    elif config['preprocessor_pipeline']['sampling'] == "undersample":
        sampler_parameters = dict(config['undersample_parameters'])
        sampler = RandomUnderSampler(**sampler_parameters)

    if config['preprocessor_pipeline']['scaling'] == "standardscaler":
        scaler = StandardScaler()

    elif config['preprocessor_pipeline']['scaling'] == "minmax":
        scaler = MinMaxScaler()

    preprocessor_scaler, preprocessor_sampler = ('scaler', scaler), ('sampler', sampler)

    return preprocessor_scaler, preprocessor_sampler


def create_model_pipeline(config):

    if config['model']['model'] == "logreg":
        param_grid = dict(config['logreg_hyperparameters'])
        classifier = LogisticRegression(**param_grid)

    elif config['model']['model'] == "knn":
        param_grid = dict(config['knn_hyperparameters'])
        classifier = KNeighborsClassifier(**param_grid)

    elif config['model']['model'] == "rfc":
        param_grid = dict(config['rfc_hyperparameters'])
        classifier = RandomForestClassifier(**param_grid)

    elif config['model']['model'] == "gbc":
        param_grid = dict(config['gbc_hyperparameters'])
        classifier = GradientBoostingClassifier(**param_grid)

    return classifier

def create_cross_validation_pipeline(config):

    if config['cross_validation']['cv'] == "kfold":
        cv_parameters = dict(config['kfold_parameters'])
        cv = KFold(**cv_parameters)

    elif config['cross_validation']['cv'] == "stratkfold":
        cv_parameters = dict(config['stratkfold_parameters'])
        cv = StratifiedKFold(**cv_parameters)
    
    elif config['cross_validation']['cv'] == "repeatstratkfold":
        cv_parameters = dict(config['repeatstratkfold_parameters'])
        cv = RepeatedStratifiedKFold(**cv_parameters)

    return cv

def scoring_metric(config):

    if config['scoring_metric']['scoring'] == "accuracy":
        scoring = "accuracy"

    elif config['scoring_metric']['scoring'] == "precision":
        scoring = "precision"

    elif config['scoring_metric']['scoring'] == "recall":
        scoring = "recall"

    elif config['scoring_metric']['scoring'] == "f1":
        scoring = "f1"

    return scoring