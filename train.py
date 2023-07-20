import warnings
warnings.filterwarnings("ignore")

import numpy as np
import argparse
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline

import data_preprocessing
import training_pipeline


class train_validate(object):
    def __init__(self, arg):
        self.arg = arg
        self._pipeline = None
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

    def train(self, config):

        X, y = data_preprocessing.main()

        # Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2023, stratify=y)
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        preprocessor_scaler, preprocessor_sampler = training_pipeline.create_processing_pipeline(config)
        classifier = training_pipeline.create_model_pipeline(config)
        cv = training_pipeline.create_cross_validation_pipeline(config)

        steps = [preprocessor_scaler, preprocessor_sampler, ('classifier', classifier)]

        pipeline = Pipeline(steps=steps)

        scoring_metrics = ['accuracy','precision','recall','f1']

        print('\n',"=======================================================",'\n')
        print("Model:", '\n', classifier)
        print("_______________________________________________________")
        print('\n',"Training Evaluation Metrics:", '\n')
        for metric in scoring_metrics:
            scores = cross_val_score(pipeline, X_train, y_train, scoring=metric, cv=cv, n_jobs=-1)
            score = np.mean(scores)
            print(metric, 'Score: %.3f' % score)
        print("_______________________________________________________")

        self._pipeline = pipeline
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

    def validate(self):
        self._pipeline.fit(self._X_train, self._y_train)
        y_hat = self._pipeline.predict(self._X_test)
        conf_mat = confusion_matrix(self._y_test, y_hat)
        class_report = classification_report(self._y_test, y_hat)
        print('\n',"Validation Evaluation Metrics:", '\n')
        print("Confusion Matrix")
        print(conf_mat, '\n')
        print("Classification Report:")
        print(class_report)
        print('\n',"=======================================================",'\n')


if __name__== '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config",
                        help="Absolute path to configuration file.")
    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No configuration file provided.")
        exit()

    else:
        with open(args.config, "r") as inp:
            config = json.load(inp)

    train_validate = train_validate(config)
    train = train_validate.train(config)
    validate = train_validate.validate()

