"""
Code adapt from https://github.com/rwightman/pytorch-image-models
"""
from copy import deepcopy
import os
import logging
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Dict
from itertools import chain
from torch.utils.checkpoint import checkpoint

# Global variables for rarely used pretrained checkpoint download progress and hash check.
# Use set_pretrained_download_progress / set_pretrained_check_hash functions to toggle.
_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False

_model_pretrained_cfgs = dict()  # central repo for model default_cfgs
_logger = logging.getLogger(__name__)

try:
    from huggingface_hub import HfApi, HfFolder, Repository, cached_download, hf_hub_url
    cached_download = partial(cached_download, library_name="timm", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    cached_download = None
    _has_hf_hub = False

try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

def get_pretrained_cfg(model_name):
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])
    return {}

def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub

def resolve_pretrained_cfg(variant: str, pretrained_cfg=None, kwargs=None):
    if pretrained_cfg and isinstance(pretrained_cfg, dict):
        # highest priority, pretrained_cfg available and passed explicitly
        return deepcopy(pretrained_cfg)
    if kwargs and 'pretrained_cfg' in kwargs:
        # next highest, pretrained_cfg in a kwargs dict, pop and return
        pretrained_cfg = kwargs.pop('pretrained_cfg', {})
        if pretrained_cfg:
            return deepcopy(pretrained_cfg)
    # lookup pretrained cfg in model registry by variant
    pretrained_cfg = get_pretrained_cfg(variant)
    assert pretrained_cfg
    return pretrained_cfg

def set_default_kwargs(kwargs, names, pretrained_cfg):
    for n in names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

def filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)

def update_pretrained_cfg_and_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)
    set_default_kwargs(kwargs, names=default_kwarg_names, pretrained_cfg=pretrained_cfg)
    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    filter_kwargs(kwargs, names=kwargs_filter)

def _resolve_pretrained_source(pretrained_cfg):
    cfg_source = pretrained_cfg.get('source', '')
    pretrained_url = pretrained_cfg.get('url', None)
    pretrained_file = pretrained_cfg.get('file', None)
    hf_hub_id = pretrained_cfg.get('hf_hub_id', None)
    # resolve where to load pretrained weights from
    load_from = ''
    pretrained_loc = ''
    if cfg_source == 'hf-hub' and has_hf_hub(necessary=True):
        # hf-hub specified as source via model identifier
        load_from = 'hf-hub'
        assert hf_hub_id
        pretrained_loc = hf_hub_id
    else:
        # default source == timm or unspecified
        if pretrained_file:
            load_from = 'file'
            pretrained_loc = pretrained_file
        elif pretrained_url:
            load_from = 'url'
            pretrained_loc = pretrained_url
        elif hf_hub_id and has_hf_hub(necessary=False):
            # hf-hub available as alternate weight source in default_cfg
            load_from = 'hf-hub'
            pretrained_loc = hf_hub_id
    return load_from, pretrained_loc


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        pretrained_strict: bool = True,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        pretrained_cfg (dict): model's pretrained weight/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        pretrained_custom_load (bool): use custom load fn, to load numpy or other non PyTorch weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = kwargs.pop('pretrained_cfg', {})
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=pretrained_cfg)
    update_pretrained_cfg_and_kwargs(pretrained_cfg, kwargs, kwargs_filter)
    pretrained_cfg.setdefault('architecture', variant)

    # Build the model
    model = model_cls(**kwargs) #if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    # model.fc = nn.Linear(model.fc.in_features, 2)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    # num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000)) # features = false
    num_classes_pretrained = getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:

        load_pretrained(
            model,
            pretrained_cfg=pretrained_cfg,
            num_classes=num_classes_pretrained,
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=None,
            strict=pretrained_strict)

    return model


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict

def load_state_dict(checkpoint_path, use_ema=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        num_classes (int): num_classes for model
        in_chans (int): in_chans for model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint

    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None) or {}
    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        state_dict = load_state_dict_from_url(
            pretrained_loc, map_location='cpu', progress=_DOWNLOAD_PROGRESS, check_hash=_CHECK_HASH)
    # elif load_from == 'hf-hub':
    #     _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
    #     state_dict = load_state_dict_from_hf(pretrained_loc) ## assume SKnet not load from hf-hub
    else:
        _logger.warning("No pretrained weights exist or were found for this model. Using random initialization.")
        return

    # if filter_fn is not None: ## it is NONE
    #     # for backwards compat with filter fn that take one arg, try one first, the two
    #     try:
    #         state_dict = filter_fn(state_dict)
    #     except TypeError:
    #         state_dict = filter_fn(state_dict, model)

    input_convs = pretrained_cfg.get('first_conv', None)
    # if input_convs is not None and in_chans != 3:  # in_chans is indeed 3
    #     if isinstance(input_convs, str):
    #         input_convs = (input_convs,)
    #     for input_conv_name in input_convs:
    #         weight_name = input_conv_name + '.weight'
    #         try:
    #             state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
    #             _logger.info(
    #                 f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
    #         except NotImplementedError as e:
    #             del state_dict[weight_name]
    #             strict = False
    #             _logger.warning(
    #                 f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    model.load_state_dict(state_dict, strict=strict)

def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x
