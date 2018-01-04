from visdom import Visdom
import numpy as np


class Dashboard(Visdom):
    def __init__(self, server=None, port=None, env='main'):
        super(Dashboard, self).__init__(server=server, port=port, env=env)
        self.plots = {}

    def plot(self, name, type, *args, **kwargs):
        if 'opts' not in kwargs:
            kwargs['opts'] = {}
        if 'title' not in kwargs['opts']:
            kwargs['opts']['title'] = name

        if hasattr(self, type):
            if name in self.plots:
                getattr(self, type)(win=self.plots[name], *args, **kwargs)
            else:
                id = getattr(self, type)(*args, **kwargs)
                self.plots[name] = id
        else:
            raise AttributeError('plot type: {} does not exist. Please'
                                 'refer to visdom documentation.'.format(type))

    def append(self, name, type, *args, **kwargs):
        if name in self.plots:
            if type == 'image' or type == 'images':
                self.plot(name, type, *args, **kwargs)
            else:
                self.plot(name, type, *args, update='replace', **kwargs)
        else:
            self.plot(name, type, *args, **kwargs)

    def remove(self, name):
        del self.plots[name]

    def clear(self):
        self.plots = {}

    def draw(self, train_data, val_data, datatype):
        self.append(name=datatype, type='line',
                    X=np.stack((np.array(list(train_data.keys())),
                               np.array(list(val_data.keys())))).transpose(),
                    Y=np.stack((np.array(list(train_data.values())),
                               np.array(list(val_data.values())))).transpose(),
                    opts=dict(legend=['train_{}'.format(datatype), 'val_{}'.format(datatype)],
                              update=True,
                              xlabel='Epoch',
                              ylabel=datatype))

    def draw_batch_curve(self, train_data, val_data, datatype):

        diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
                    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia']

        # get the data from dict
        epoch = np.array(list(train_data.keys()))
        training_data = np.array(list(train_data.values()))
        validation_data = np.array(list(val_data.values()))

        for idx, disease in enumerate(diseases):
            train_idx = training_data[:,idx]
            val_idx = validation_data[:,idx]
            self.append(name='{}/{}'.format(disease,datatype), type='line',
                        X=np.stack((epoch,epoch)).transpose(),
                        Y=np.stack((train_idx,val_idx)).transpose(),
                        opts=dict(legend=['{}/train'.format(datatype), '{}/val'.format(datatype)],
                                  update=True,
                                  xlabel='Epoch',
                                  ylabel=datatype,
                                  ))