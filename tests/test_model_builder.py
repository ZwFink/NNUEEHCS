import pytest
from nnueehcs.model_builder import (build_network, ModelBuilder,
                                    DeltaUQMLPModelBuilder,
                                    EnsembleModelBuilder,
                                    PAGERModelBuilder,
                                    KDEModelBuilder,
                                    KDEMLPModel,
                                    MCDropoutModelBuilder,
                                    expand_repeats)
import torch
import io
import yaml
import os
from torch import nn
from deltauq import deltaUQ_MLP, deltaUQ_CNN

def assert_models_equal(model1, model2):
    for layer1, layer2 in zip(model1.children(), model2.children()):
        assert type(layer1) == type(layer2), f"Layer types differ: {type(layer1)} != {type(layer2)}"
        
        if isinstance(layer1, nn.Conv2d):
            assert layer1.in_channels == layer2.in_channels, f"In channels differ: {layer1.in_channels} != {layer2.in_channels}"
            assert layer1.out_channels == layer2.out_channels, f"Out channels differ: {layer1.out_channels} != {layer2.out_channels}"
            assert layer1.kernel_size == layer2.kernel_size, f"Kernel sizes differ: {layer1.kernel_size} != {layer2.kernel_size}"
            assert layer1.stride == layer2.stride, f"Strides differ: {layer1.stride} != {layer2.stride}"
            assert layer1.padding == layer2.padding, f"Padding differs: {layer1.padding} != {layer2.padding}"
        
        elif isinstance(layer1, nn.Linear):
            assert layer1.in_features == layer2.in_features, f"In features differ: {layer1.in_features} != {layer2.in_features}"
            assert layer1.out_features == layer2.out_features, f"Out features differ: {layer1.out_features} != {layer2.out_features}"
        
        elif isinstance(layer1, nn.BatchNorm2d):
            assert layer1.num_features == layer2.num_features, f"Num features differ: {layer1.num_features} != {layer2.num_features}"
        
        elif isinstance(layer1, (nn.ReLU, nn.Dropout)):
            assert layer1.inplace == layer2.inplace, f"Inplace flag differs: {layer1.inplace} != {layer2.inplace}"
        
        if isinstance(layer1, nn.Dropout):
            assert layer1.p == layer2.p, f"Dropout probability differs: {layer1.p} != {layer2.p}"

@pytest.fixture()
def architecture1():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=25, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 25, kernel_size=5, stride=1, padding='same')
    )
    return model

@pytest.fixture()
def architecture2():
    model = nn.Sequential(
        nn.Linear(16, 25),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(25),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(25, 5)
    )
    return model


@pytest.fixture()
def kde_architecture1(architecture1):
    return KDEMLPModel(architecture1)


@pytest.fixture()
def kde_architecture2(architecture2):
    return KDEMLPModel(architecture2)


@pytest.fixture()
def duq_architecture1():
    model = nn.Sequential(
        nn.Conv2d(6, 16, kernel_size=25, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 25, kernel_size=5, stride=1, padding='same')
    )
    return deltaUQ_CNN(model)


@pytest.fixture()
def duq_architecture2():
    model = nn.Sequential(
        nn.Linear(32, 25),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(25),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(25, 5)
    )
    return deltaUQ_MLP(model)

@pytest.fixture()
def model_descr_yaml():
    model_yaml = """
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
        padding: same

architecture2:
    - Linear:
        args: [16, 25]
    - ReLU:
        inplace: true
    - BatchNorm2d:
        args: [25]
    - Dropout:
        args: [0.2]
        inplace: True
    - Linear:
        args: [25, 5]

architecture3:
    - Linear:
        args: [16, 25]
    - ReLU:
        inplace: true
    - Linear:
        args: [25, 25]
    - ReLU:
        inplace: true
    - Linear:
        args: [25, 25]
    - ReLU:
        inplace: true
    - Linear:
        args: [25, 5]
delta_uq_model:
    estimator: std
    num_anchors: 2
mc_dropout_model:
    num_samples: 10
    dropout_percent: 0.2
pager_model:
    estimator: std
    num_anchors: 3
kde_model:
    bandwidth: scott
ensemble_model:
    num_models: 10
"""
    return model_yaml


def test_build_network(model_descr_yaml, architecture1, architecture2):

    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    
    # Test architecture
    actual_model = build_network(model_descr['architecture'])
    assert_models_equal(actual_model, architecture1)

    # Test architecture2
    actual_model2 = build_network(model_descr['architecture2'])
    assert_models_equal(actual_model2, architecture2)


def test_model_builder(model_descr_yaml, architecture1, architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = ModelBuilder(model_descr['architecture'])
    arch1 = model_builder.build()
    assert_models_equal(arch1, architecture1)
    builder2 = ModelBuilder(model_descr['architecture2'])
    arch2 = builder2.build()
    assert_models_equal(arch2, architecture2)

    info = model_builder.get_info()
    assert info.is_cnn() == True
    assert info.is_mlp() == False
    assert info.num_layers() == 3
    assert info.num_inputs() == 3

    info2 = builder2.get_info()
    assert info2.is_cnn() == False
    assert info2.is_mlp() == True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 16

    assert not hasattr(info, 'get_estimator')


def test_duq_model_builder(model_descr_yaml, duq_architecture1, duq_architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = DeltaUQMLPModelBuilder(model_descr['architecture'], model_descr['delta_uq_model'])
    info = model_builder.get_info()
    assert info.is_cnn() == True
    assert info.is_mlp() == False
    assert info.num_layers() == 3
    assert info.num_inputs() == 6
    assert info.get_estimator() == 'std'

    net = model_builder.build()
    assert_models_equal(net, duq_architecture1)
    assert net.num_anchors == 2

    model_builder = DeltaUQMLPModelBuilder(model_descr['architecture2'], model_descr['delta_uq_model'])
    info = model_builder.get_info()
    assert info.is_cnn() == False
    assert info.is_mlp() == True
    assert info.num_layers() == 5
    assert info.num_inputs() == 32
    assert info.get_estimator() == 'std'

    net = model_builder.build()
    assert_models_equal(net, duq_architecture2)


def test_pager_model_builder(model_descr_yaml, duq_architecture1, duq_architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = PAGERModelBuilder(model_descr['architecture'], model_descr['pager_model'])
    net = model_builder.build()
    assert net.num_anchors == 3
    assert net.num_anchors == 3
    assert_models_equal(net, duq_architecture1)
    info = model_builder.get_info()
    assert info.is_cnn() is True
    assert info.is_mlp() is False
    assert info.num_layers() == 3
    assert info.num_inputs() == 6
    assert info.get_estimator() == 'std'

    model_builder2 = PAGERModelBuilder(model_descr['architecture2'], model_descr['pager_model'])
    net2 = model_builder2.build()
    assert_models_equal(net2, duq_architecture2)
    info2 = model_builder2.get_info()
    assert info2.is_cnn() is False
    assert info2.is_mlp() is True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 32
    assert info2.get_estimator() == 'std'


def test_ensemble_model_builder(model_descr_yaml):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = EnsembleModelBuilder(model_descr['architecture'], model_descr['ensemble_model'])
    ensemble = model_builder.build()
    info = model_builder.get_info()
    assert info.get_num_models() == 10

    assert info.is_cnn() == True
    assert info.is_mlp() == False
    assert info.num_layers() == 3
    assert info.num_inputs() == 3

    assert not hasattr(info, 'get_estimator')

    model_builder2 = EnsembleModelBuilder(model_descr['architecture2'], model_descr['ensemble_model'])
    ensemble2 = model_builder2.build()
    info2 = model_builder2.get_info()
    assert info2.get_num_models() == 10

    assert info2.is_cnn() == False
    assert info2.is_mlp() == True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 16

    assert not hasattr(info2, 'get_estimator')


def test_kde_model_builder(model_descr_yaml, kde_architecture1, kde_architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = KDEModelBuilder(model_descr['architecture'],
                                    model_descr['kde_model'])
    arch1 = model_builder.build()
    assert_models_equal(arch1, kde_architecture1)
    builder2 = KDEModelBuilder(model_descr['architecture2'],
                               model_descr['kde_model'])
    arch2 = builder2.build()
    assert_models_equal(arch2, kde_architecture2)

    info = model_builder.get_info()
    assert info.is_cnn() is True
    assert info.is_mlp() is False
    assert info.num_layers() == 3
    assert info.num_inputs() == 3

    info2 = builder2.get_info()
    assert info2.is_cnn() is False
    assert info2.is_mlp() is True
    assert info2.num_layers() == 5
    assert info2.num_inputs() == 16

    assert not hasattr(info, 'get_estimator')

def test_mc_model_builder(model_descr_yaml, architecture1, architecture2):
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    model_builder = MCDropoutModelBuilder(model_descr['architecture3'],
                                          model_descr['mc_dropout_model'])
    arch1 = model_builder.build()
    expected = """
MCDropoutModel(
  (model): Sequential(
    (0): Linear(in_features=16, out_features=25, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=25, out_features=25, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=25, out_features=25, bias=True)
    (7): ReLU(inplace=True)
    (8): Linear(in_features=25, out_features=5, bias=True)
  )
)
"""
    assert str(arch1).strip() == expected.strip()
    arch1.eval()
    for layer in arch1.model:
        if isinstance(layer, nn.Dropout):
            assert layer.training == True
        else:
            assert layer.training == False
    arch1.train()
    for layer in arch1.model:
        assert layer.training == True

    model_builder = MCDropoutModelBuilder(model_descr['architecture2'],
                                          model_descr['mc_dropout_model'])
    arch2 = model_builder.build()
    arch2.eval()
    for layer in arch2.model:
        if isinstance(layer, nn.Dropout):
            assert layer.training == True
        else:
            assert layer.training == False
    arch2.train()
    for layer in arch2.model:
        assert layer.training == True


def test_expand_repeats_basic():
    """Test basic repeat functionality"""
    architecture = [
        {'Linear': {'args': [4, 64]}},
        {'Repeat': {
            'count': 3,
            'layers': [
                {'Linear': {'args': [64, 64]}},
                {'ReLU': {'inplace': True}}
            ]
        }},
        {'Linear': {'args': [64, 1]}}
    ]
    
    expanded = expand_repeats(architecture)
    
    # Should have: 1 initial Linear + (3 * 2 repeated layers) + 1 final Linear = 8 layers
    expected_length = 1 + (3 * 2) + 1
    assert len(expanded) == expected_length
    
    # Check structure
    assert 'Linear' in expanded[0]
    assert expanded[0]['Linear']['args'] == [4, 64]
    
    # Check repeated layers
    for i in range(1, 7, 2):  # indices 1, 3, 5 should be Linear layers
        assert 'Linear' in expanded[i]
        assert expanded[i]['Linear']['args'] == [64, 64]
    
    for i in range(2, 8, 2):  # indices 2, 4, 6 should be ReLU layers
        assert 'ReLU' in expanded[i]
        assert expanded[i]['ReLU']['inplace'] == True
    
    # Check final layer
    assert 'Linear' in expanded[7]
    assert expanded[7]['Linear']['args'] == [64, 1]


def test_expand_repeats_nested():
    """Test nested repeat functionality"""
    architecture = [
        {'Linear': {'args': [4, 32]}},
        {'Repeat': {
            'count': 2,
            'layers': [
                {'Linear': {'args': [32, 32]}},
                {'Repeat': {
                    'count': 2,
                    'layers': [
                        {'BatchNorm1d': {'args': [32]}},
                        {'ReLU': {'inplace': True}}
                    ]
                }}
            ]
        }},
        {'Linear': {'args': [32, 1]}}
    ]
    
    expanded = expand_repeats(architecture)
    
    # Should have: 1 initial + 2 * (1 Linear + 2 * (1 BatchNorm + 1 ReLU)) + 1 final
    # = 1 + 2 * (1 + 4) + 1 = 1 + 10 + 1 = 12 layers
    expected_length = 12
    assert len(expanded) == expected_length


def test_build_network_with_repeat():
    """Test that build_network works with repeat blocks"""
    architecture = [
        {'Linear': {'args': [4, 32]}},
        {'Repeat': {
            'count': 2,
            'layers': [
                {'Linear': {'args': [32, 32]}},
                {'ReLU': {'inplace': True}}
            ]
        }},
        {'Linear': {'args': [32, 1]}}
    ]
    
    # This should not raise an exception
    network = build_network(architecture)
    
    # Check that we can create a model instance
    assert isinstance(network, torch.nn.Sequential)
    assert len(network) == 6  # 1 + 2*2 + 1 = 6 layers
    
    # Test with dummy input
    dummy_input = torch.randn(10, 4)
    output = network(dummy_input)
    assert output.shape == (10, 1)


def test_repeat_equivalence():
    """Test that repeat blocks produce the same result as manually expanded architectures"""
    # Original architecture (expanded manually)
    original_arch = [
        {'Linear': {'args': [4, 2048]}},
        {'BatchNorm1d': {'args': [2048]}},
        {'ReLU': {'inplace': True}},
        {'Linear': {'args': [2048, 2048]}},
        {'BatchNorm1d': {'args': [2048]}},
        {'ReLU': {'inplace': True}},
        {'Linear': {'args': [2048, 2048]}},
        {'BatchNorm1d': {'args': [2048]}},
        {'ReLU': {'inplace': True}},
        {'Linear': {'args': [2048, 1]}}
    ]
    
    # New architecture with repeat
    repeat_arch = [
        {'Linear': {'args': [4, 2048]}},
        {'BatchNorm1d': {'args': [2048]}},
        {'ReLU': {'inplace': True}},
        {'Repeat': {
            'count': 2,
            'layers': [
                {'Linear': {'args': [2048, 2048]}},
                {'BatchNorm1d': {'args': [2048]}},
                {'ReLU': {'inplace': True}}
            ]
        }},
        {'Linear': {'args': [2048, 1]}}
    ]
    
    # Expand the repeat architecture
    expanded_repeat = expand_repeats(repeat_arch)
    
    # They should be equivalent
    assert len(expanded_repeat) == len(original_arch)
    
    for i, (orig, new) in enumerate(zip(original_arch, expanded_repeat)):
        assert orig == new, f"Layer {i} differs: {orig} vs {new}"


def test_repeat_yaml_integration():
    """Test repeat functionality with YAML configuration"""
    yaml_config = """
architecture_with_repeat:
    - Linear:
        args: [10, 50]
    - Repeat:
        count: 3
        layers:
          - Linear:
              args: [50, 50]
          - ReLU:
              inplace: true
          - BatchNorm1d:
              args: [50]
    - Linear:
        args: [50, 1]
"""
    
    config = yaml.safe_load(io.StringIO(yaml_config))
    architecture = config['architecture_with_repeat']
    
    # Build network should work with repeat blocks
    network = build_network(architecture)
    
    # Should have: 1 + 3*3 + 1 = 11 layers
    assert len(network) == 11
    
    # Test that it produces valid output
    dummy_input = torch.randn(5, 10)
    output = network(dummy_input)
    assert output.shape == (5, 1)


def test_repeat_empty_layers():
    """Test edge case with empty layers in repeat block"""
    architecture = [
        {'Linear': {'args': [4, 32]}},
        {'Repeat': {
            'count': 0,
            'layers': [
                {'Linear': {'args': [32, 32]}},
                {'ReLU': {'inplace': True}}
            ]
        }},
        {'Linear': {'args': [32, 1]}}
    ]
    
    expanded = expand_repeats(architecture)
    
    # Should have only the first and last layers
    assert len(expanded) == 2
    assert 'Linear' in expanded[0]
    assert expanded[0]['Linear']['args'] == [4, 32]
    assert 'Linear' in expanded[1]
    assert expanded[1]['Linear']['args'] == [32, 1]

def test_info_grabber_outputs(model_descr_yaml):
    """Test num_outputs and set_num_outputs methods for both CNN and MLP InfoGrabbers"""
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    
    # Test MLPInfoGrabber outputs
    builder = ModelBuilder(model_descr['architecture2'])
    info = builder.get_info()
    assert info.is_mlp() == True
    assert info.num_outputs() == 5  # Last Linear layer outputs 5
    
    # Test setting outputs
    info.set_num_outputs(10)
    assert info.num_outputs() == 10
    
    # Verify the underlying description was updated
    last_linear_found = False
    for layer_dict in reversed(info.descr):
        if 'Linear' in layer_dict:
            assert layer_dict['Linear']['args'][1] == 10
            last_linear_found = True
            break
    assert last_linear_found, "Should have found a Linear layer"
    
    # Test CNNInfoGrabber outputs
    builder_cnn = ModelBuilder(model_descr['architecture'])
    info_cnn = builder_cnn.get_info()
    assert info_cnn.is_cnn() == True
    assert info_cnn.num_outputs() == 25  # Last Conv2d layer outputs 25 channels
    
    # Test setting outputs for CNN
    info_cnn.set_num_outputs(64)
    assert info_cnn.num_outputs() == 64
    
    # Verify the underlying description was updated
    last_conv_found = False
    for layer_dict in reversed(info_cnn.descr):
        if 'Conv2d' in layer_dict:
            assert layer_dict['Conv2d']['args'][1] == 64
            last_conv_found = True
            break
    assert last_conv_found, "Should have found a Conv2d layer"


def test_cnn_with_final_linear_outputs():
    """Test CNNInfoGrabber with architecture that ends in Linear layer"""
    cnn_with_linear_arch = [
        {'Conv2d': {'args': [3, 16, 5], 'stride': 1, 'padding': 2}},
        {'ReLU': {'inplace': True}},
        {'Linear': {'args': [256, 10]}}  # Final linear layer
    ]
    
    builder = ModelBuilder(cnn_with_linear_arch)
    info = builder.get_info()
    assert info.is_cnn() == True
    assert info.num_outputs() == 10  # Should find the Linear layer output
    
    # Test setting outputs
    info.set_num_outputs(20)
    assert info.num_outputs() == 20
    
    # Verify the Linear layer was updated in the InfoGrabber's description
    # Note: the InfoGrabber works on a copy of the architecture, not the original
    assert info.descr[2]['Linear']['args'][1] == 20
    assert info.descr[0]['Conv2d']['args'][1] == 16  # Should be unchanged
    
    # Original architecture should remain unchanged since ModelBuilder makes a copy
    assert cnn_with_linear_arch[2]['Linear']['args'][1] == 10
    assert cnn_with_linear_arch[0]['Conv2d']['args'][1] == 16


def test_outputs_with_various_architectures():
    """Test output methods with different architecture patterns"""
    
    # Test architecture with multiple Linear layers
    multi_linear_arch = [
        {'Linear': {'args': [10, 50]}},
        {'ReLU': {'inplace': True}},
        {'Linear': {'args': [50, 25]}},
        {'ReLU': {'inplace': True}},
        {'Linear': {'args': [25, 1]}}  # This should be the output layer
    ]
    
    builder = ModelBuilder(multi_linear_arch)
    info = builder.get_info()
    assert info.num_outputs() == 1
    
    info.set_num_outputs(5)
    assert info.num_outputs() == 5
    # Check InfoGrabber's internal description, not the original
    assert info.descr[4]['Linear']['args'][1] == 5  # Last layer should be updated
    assert info.descr[2]['Linear']['args'][1] == 25  # Previous layers unchanged
    assert info.descr[0]['Linear']['args'][1] == 50
    
    # Test architecture with only non-Linear/Conv2d layers at the end
    non_output_arch = [
        {'Linear': {'args': [10, 5]}},
        {'ReLU': {'inplace': True}},
        {'Dropout': {'args': [0.5]}}
    ]
    
    builder2 = ModelBuilder(non_output_arch)
    info2 = builder2.get_info()
    assert info2.num_outputs() == 5  # Should find the Linear layer
    
    info2.set_num_outputs(8)
    assert info2.num_outputs() == 8
    # Check InfoGrabber's internal description
    assert info2.descr[0]['Linear']['args'][1] == 8


def test_deep_evidential_uses_set_num_outputs(model_descr_yaml):
    """Test that DeepEvidentialModelBuilder uses the set_num_outputs method"""
    from nnueehcs.model_builder import DeepEvidentialModelBuilder
    
    model_descr = yaml.safe_load(io.StringIO(model_descr_yaml))
    der_config = {'some_param': 'value'}  # Minimal config for testing
    
    # Test with MLP architecture
    builder = DeepEvidentialModelBuilder(model_descr['architecture2'], der_config)
    
    # Check original architecture before any modifications
    assert model_descr['architecture2'][4]['Linear']['args'][1] == 5
    
    info = builder.get_info()
    
    assert info.num_outputs() == 9
    
    # Verify the last Linear layer was properly updated
    last_linear_found = False
    for layer_dict in reversed(info.descr):
        if 'Linear' in layer_dict:
            assert layer_dict['Linear']['args'][1] == 9
            last_linear_found = True
            break
    assert last_linear_found


def test_outputs_edge_cases():
    """Test edge cases for output methods"""
    
    # Test architecture with no Linear or Conv2d layers
    no_output_arch = [
        {'ReLU': {'inplace': True}},
        {'Dropout': {'args': [0.5]}}
    ]
    
    builder = ModelBuilder(no_output_arch)
    info = builder.get_info()  # This will create MLPInfoGrabber by default
    assert info.num_outputs() is None  # Should return None when no suitable layer found
    
    # set_num_outputs should do nothing when no suitable layer is found
    info.set_num_outputs(10)
    assert info.num_outputs() is None  # Should still be None