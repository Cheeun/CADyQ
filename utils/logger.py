#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import pandas as pd
import pdb
import os

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, csv_path='results.csv', log_path=None, resume=False): 
        self.path = csv_path
        self.figures = []
        self.results = None

    def add(self, logger_dict):
        df = pd.DataFrame([logger_dict.values()], columns=logger_dict.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results = pd.read_csv(path)

    def mask_log(self):
        return logging


