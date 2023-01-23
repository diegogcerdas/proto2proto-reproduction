import torch
import torch.nn as nn
import torch.nn.functional as F
from .features import init_backbone
from .protopnet.model import PPNet, PPNet2
from .protopnet.receptive_field import compute_proto_layer_rf_info_v2

_prototypical_model = {

    "protopnet": PPNet
}


def init_proto_model(manager, classes, backbone):
    """
        Create network with pretrained features and 1x1 convolutional layer

    """
    # Creating tree (backbone+add-on+prototree) architecture

    prototypical_model = backbone.prototypicalModel
    use_chkpt_opt = manager.settingsConfig.useCheckpointOptimizer

    features, trainable_param_names = init_backbone(backbone)

    if prototypical_model == 'protopnet_new':

        img_size = 224
        prototype_shape = (2000, 128, 1, 1)
        num_classes = 200
        prototype_activation_function = 'log'
        add_on_layers_type='other'

        layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
        proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                                layer_filter_sizes=layer_filter_sizes,
                                                                layer_strides=layer_strides,
                                                                layer_paddings=layer_paddings,
                                                                prototype_kernel_size=prototype_shape[2])

        model =  PPNet2(
            features=features,
            img_size=img_size,
            prototype_shape=prototype_shape,
            proto_layer_rf_info=proto_layer_rf_info,
            num_classes=num_classes,
            init_weights=True,
            prototype_activation_function=prototype_activation_function,
            add_on_layers_type=add_on_layers_type
        )
        model.load_state_dict(torch.load(backbone.loadPath, map_location=torch.device('cpu')))

        checkpoint = None

    else:

        model = _prototypical_model[prototypical_model](
            num_classes=len(classes), feature_net=features, args=manager.settingsConfig)

        if backbone.loadPath is not None:
            checkpoint = torch.load(backbone.loadPath, map_location=torch.device('cpu'))
            model = checkpoint['model']
            print("Loaded model from ", backbone.loadPath)

            if not use_chkpt_opt:
                checkpoint = None
        else:
            checkpoint = None

    if manager.common.mgpus:
        print("Multi-gpu setting")
        model = nn.DataParallel(model)

    if manager.common.cuda > 0:
        print("Using GPU")
        model.cuda()

    return model, checkpoint, trainable_param_names
