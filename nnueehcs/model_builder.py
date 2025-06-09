import torch.nn
import collections
import io
import yaml
from .models import MLPModel, KDEMLPModel, DeltaUQMLP, EnsembleModel, PAGERMLP, MCDropoutModel, KNNKDEMLPModel, BaselineModel, DeepEvidentialModel
import copy
import types


class LayerBuilder(object):
    # adapted from https://gist.github.com/ferrine/89d739e80712f5549e44b2c2435979ef
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)


def expand_repeats(architecture):
    """
    Expand Repeat blocks in the architecture into their full form.
    
    Args:
        architecture (list): List of layer dictionaries, potentially containing Repeat blocks
        
    Returns:
        list: Expanded architecture with Repeat blocks replaced by their repeated content
        
    Example:
        Input:
        [
            {'Linear': {'args': [4, 2048]}},
            {'Repeat': {'count': 3, 'layers': [
                {'Linear': {'args': [2048, 2048]}},
                {'ReLU': {'inplace': True}}
            ]}},
            {'Linear': {'args': [2048, 1]}}
        ]
        
        Output:
        [
            {'Linear': {'args': [4, 2048]}},
            {'Linear': {'args': [2048, 2048]}},
            {'ReLU': {'inplace': True}},
            {'Linear': {'args': [2048, 2048]}},
            {'ReLU': {'inplace': True}},
            {'Linear': {'args': [2048, 2048]}},
            {'ReLU': {'inplace': True}},
            {'Linear': {'args': [2048, 1]}}
        ]
    """
    expanded = []
    
    for block in architecture:
        if 'Repeat' in block:
            repeat_config = block['Repeat']
            count = repeat_config['count']
            layers_to_repeat = repeat_config['layers']
            
            expanded_layers = expand_repeats(layers_to_repeat)
            
            # Add the expanded layers 'count' times
            for _ in range(count):
                expanded.extend(copy.deepcopy(expanded_layers))
        else:
            expanded.append(block)
    
    return expanded


def build_network(architecture, builder=LayerBuilder(torch.nn.__dict__)):
    """
    Configuration for feedforward networks is list by nature. We can write 
    this in simple data structures. In yaml format it can look like:
    .. code-block:: yaml
        architecture:
            - Conv2d:
                args: [3, 16, 25]
                stride: 1
                padding: 2
            - ReLU:
                inplace: true
            - Conv2d:
                args: [16, 25, 5]
                stride: 1
                padding: 2
            - Repeat:
                count: 3
                layers:
                  - Linear:
                      args: [256, 256]
                  - ReLU:
                      inplace: true
    Note, that each layer is a list with a single dict, this is for readability.
    For example, `builder` for the first block is called like this:
    .. code-block:: python
        first_layer = builder("Conv2d", *[3, 16, 25], **{"stride": 1, "padding": 2})
    the simpliest ever builder is just the following function:
    .. code-block:: python
         def build_layer(name, *args, **kwargs):
            return layers_dictionary[name](*args, **kwargs)
    
    Some more advanced builders catch exceptions and format them in debuggable way or merge 
    namespaces for name lookup
    
    .. code-block:: python
    
        extended_builder = Builder(torch.nn.__dict__, mynnlib.__dict__)
        net = build_network(architecture, builder=extended_builder)
        
    """
    layers = []
    
    for block in architecture:
        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        kwargs_copy = kwargs.copy()
        args = kwargs_copy.pop("args", [])
        layers.append(builder(name, *args, **kwargs_copy))
    return torch.nn.Sequential(*layers)


class InfoGrabbBase:
    def __init__(self, descr):
        self.descr = descr

    def num_layers(self):
        return len(self.descr)


class CNNInfoGrabber(InfoGrabbBase):
    def __init__(self, descr):
        super().__init__(descr)

    def is_cnn(self):
        return True

    def is_mlp(self):
        return False

    def num_inputs(self):
        return self.descr[0]['Conv2d']['args'][0]

    def set_num_inputs(self, num_inputs):
        self.descr[0]['Conv2d']['args'][0] = num_inputs

    def num_outputs(self):
        """Get the number of outputs from the last layer"""
        for layer_dict in reversed(self.descr):
            if 'Linear' in layer_dict:
                return layer_dict['Linear']['args'][1]
            elif 'Conv2d' in layer_dict:
                return layer_dict['Conv2d']['args'][1]
        return None

    def set_num_outputs(self, num_outputs):
        """Set the number of outputs for the last layer"""
        for layer_dict in reversed(self.descr):
            if 'Linear' in layer_dict:
                layer_dict['Linear']['args'][1] = num_outputs
                return
            elif 'Conv2d' in layer_dict:
                layer_dict['Conv2d']['args'][1] = num_outputs
                return


class MLPInfoGrabber(InfoGrabbBase):
    def __init__(self, descr):
        super().__init__(descr)

    def is_mlp(self):
        return True

    def is_cnn(self):
        return False

    def num_inputs(self):
        return self.descr[0]['Linear']['args'][0]

    def set_num_inputs(self, num_inputs):
        self.descr[0]['Linear']['args'][0] = num_inputs

    def num_outputs(self):
        """Get the number of outputs from the last Linear layer"""
        for layer_dict in reversed(self.descr):
            if 'Linear' in layer_dict:
                return layer_dict['Linear']['args'][1]
        return None

    def set_num_outputs(self, num_outputs):
        """Set the number of outputs for the last Linear layer"""
        for layer_dict in reversed(self.descr):
            if 'Linear' in layer_dict:
                layer_dict['Linear']['args'][1] = num_outputs
                return


class ModelInfo:
    def __init__(self):
        pass

    @classmethod
    def get_info_grabber(cls, model_descr):
        if 'Conv2d' in model_descr[0]:
            return CNNInfoGrabber(model_descr)
        else:
            return MLPInfoGrabber(model_descr)


class ModelBuilder:
    def __init__(self, model_descr, **kwargs):
        self.model_descr = copy.deepcopy(model_descr)
        self.model_descr = expand_repeats(self.model_descr)
        if 'train_config' in kwargs:
            self.train_config = kwargs['train_config']
        else:
            self.train_config = None

    def build(self):
        built = build_network(self.model_descr)
        return built

    def update_info(self, info):
        return info

    def get_info(self):
        info = ModelInfo.get_info_grabber(self.model_descr)
        self.update_info(info)
        return info


class MLPModelBuilder(ModelBuilder):
    def __init__(self, model_descr, **kwargs):
        super().__init__(model_descr, **kwargs)

    def build(self):
        model = super().build()
        return MLPModel(model, train_config=self.train_config)


class BaselineModelBuilder(ModelBuilder):
    """Builder for BaselineModel - trains a normal network but returns dummy uncertainty values."""
    
    def __init__(self, model_descr, baseline_descr=None, **kwargs):
        super().__init__(model_descr, **kwargs)
        self.baseline_descr = baseline_descr or {}

    def build(self):
        model = super().build()
        return BaselineModel(model, 
                           train_config=self.train_config,
                           **self.baseline_descr)


class DeltaUQMLPModelBuilder(ModelBuilder):
    def __init__(self, base_descr, duq_descr, **kwargs):
        super().__init__(base_descr, **kwargs)
        self.duq_descr = duq_descr
        self._updated = False

    def build(self):
        self.update_info(self.get_info())
        base_model = super().build()
        print(base_model)
        return DeltaUQMLP(base_model, 
                          train_config=self.train_config,
                          **self.duq_descr)

    def update_info(self, info):

        estimator = self.duq_descr['estimator']
        batch_size = self.duq_descr['anchored_batch_size']

        def get_estimator(self):
            return estimator
        def get_batch_size(self):
            return batch_size
        info.get_estimator = types.MethodType(get_estimator, info)
        info.get_batch_size = types.MethodType(get_batch_size, info)
        if self._updated:
            return
        self._updated = True
        info.set_num_inputs(2 * info.num_inputs())


class PAGERModelBuilder(ModelBuilder):
    # for now, we will just inherit from DUQ.
    # Later update as needed
    def __init__(self, base_descr, pager_descr, **kwargs):
        super().__init__(base_descr, **kwargs)
        self.pager_descr = pager_descr
        self._updated = False

    def build(self):
        self.update_info(self.get_info())
        base_model = super().build()
        return PAGERMLP(base_model, 
                        train_config=self.train_config,
                        **self.pager_descr)

    def update_info(self, info):
        estimator = self.pager_descr['estimator']

        def get_estimator(self):
            return estimator
        info.get_estimator = types.MethodType(get_estimator, info)
        if self._updated:
            return
        self._updated = True
        info.set_num_inputs(2 * info.num_inputs())



class EnsembleModelBuilder(ModelBuilder):
    def __init__(self, base_descr, ensemble_descr, **kwargs):
        super().__init__(base_descr, **kwargs)
        self.ensemble_descr = ensemble_descr

    def build(self):
        info = self.get_info()
        build = super().build
        base_models = list()
        for i in range(info.get_num_models()):
            torch.manual_seed(42 + i)
            base_models.append(build())
        return EnsembleModel(base_models, train_config=self.train_config)

    def update_info(self, info):
        num_models = self.ensemble_descr['num_models']

        def get_num_models(self):
            return num_models
        info.get_num_models = types.MethodType(get_num_models, info)


class MCDropoutModelBuilder(ModelBuilder):
    def __init__(self, base_descr, dropout_descr, **kwargs):
        super().__init__(base_descr, **kwargs)
        self.dropout_descr = dropout_descr

    def build(self):
        modified_net = self._add_dropout(self.model_descr, self.dropout_descr)
        self.model_descr = modified_net
        return MCDropoutModel(super().build(),
                                 train_config=self.train_config,
                                 **self.dropout_descr
                                 )

    def _add_dropout(self, model_descr, dropout_descr):
        new_model = list()
        dropout_layer = {'Dropout': {'args': [dropout_descr['dropout_percent']]}}
        new_model.append(model_descr[0])
        for layer in model_descr[1:-1:]:
            if layer.get('Linear') or layer.get('Conv2d'):
                new_model.append(dropout_layer)
            new_model.append(layer)
        new_model.append(model_descr[-1])
        return new_model

    def update_info(self, info):
        num_samples = self.dropout_descr['num_samples']
        dropout_percent = self.dropout_descr['dropout_percent']

        def get_num_samples(self):
            return num_samples
        def get_dropout_percent(self):
            return dropout_percent
        info.get_num_samples = types.MethodType(get_num_samples, info)
        info.get_dropout_percent = types.MethodType(get_dropout_percent, info)


class KDEModelBuilder(ModelBuilder):
    def __init__(self, base_descr, kde_descr, **kwargs):
        super().__init__(base_descr, **kwargs)
        self.kde_descr = kde_descr

    def build(self):
        return KDEMLPModel(super().build(),
                           **self.kde_descr,
                           train_config=self.train_config
                           )


class KNNKDEModelBuilder(ModelBuilder):
    def __init__(self, base_descr, knn_kde_descr, **kwargs):
        super().__init__(base_descr, **kwargs)
        self.knn_kde_descr = knn_kde_descr

    def build(self):
        return KNNKDEMLPModel(super().build(), **self.knn_kde_descr, train_config=self.train_config)


class DeepEvidentialModelBuilder(ModelBuilder):
    """Builder for Deep Evidential Regression models."""
    
    def __init__(self, base_descr, der_descr, **kwargs):
        super().__init__(base_descr, **kwargs)
        self.der_descr = der_descr
        self._updated = False

    def build(self):
        # Update the architecture to output 4 additional values
        # (evidential parameters)
        self.update_info(self.get_info())
        base_model = super().build()
        return DeepEvidentialModel(base_model, 
                                 train_config=self.train_config,
                                 **self.der_descr)

    def update_info(self, info):
        """Update the last layer to output 4 values for evidential parameters"""
        if self._updated:
            return
        self._updated = True
        
        info.set_num_outputs(4 + info.num_outputs())

def get_model_builder_class(uq_method):
    """Get the appropriate model builder class for a given UQ method.
    
    Args:
        uq_method (str): The uncertainty quantification method name
        
    Returns:
        class: The corresponding model builder class
        
    Raises:
        ValueError: If the uq_method is not recognized
    """
    if uq_method == 'ensemble':
        return EnsembleModelBuilder
    elif uq_method == 'kde':
        return KDEModelBuilder
    elif uq_method == 'knn_kde':
        return KNNKDEModelBuilder
    elif uq_method == 'delta_uq':
        return DeltaUQMLPModelBuilder
    elif uq_method == 'pager':
        return PAGERModelBuilder
    elif uq_method == 'mc_dropout':
        return MCDropoutModelBuilder
    elif uq_method == 'no_uq':
        return BaselineModelBuilder
    elif uq_method == 'deep_evidential':
        return DeepEvidentialModelBuilder
    else:
        raise ValueError(f'Unknown uq method {uq_method}')