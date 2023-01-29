import torch, os, heapq, cv2
from .lib.protopnet.model import PPNet
from .lib.features import init_backbone
from .lib.utils.evaluate import acc_from_cm
from .lib.protopnet.receptive_field import compute_rf_prototype
from .lib.protopnet.find_nearest import imsave_with_bbox
from .lib.protopnet.helpers import makedir, find_high_activation_crop
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImagePatch:
    def __init__(
        self,
        patch,
        label,
        distance,
        original_img=None,
        act_pattern=None,
        patch_indices=None,
    ):
        self.patch = patch
        self.label = label
        self.negative_distance = -distance

        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


class ImagePatchInfo:
    def __init__(self, label, distance):
        self.label = label
        self.negative_distance = -distance

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


class PPNetWrapper:
    def __init__(self, args, dataloader, device):
        self.args = args
        self.dataloader = dataloader
        self.device = device
        self.model = self.load_model().to(device)

    def load_model(self):
        features, _ = init_backbone(self.args.backbone)
        model = PPNet(
            num_classes=len(self.dataloader.classes),
            feature_net=features,
            args=self.args,
        )
        state_dict = torch.load(self.args.backbone.loadPath)
        model.load_state_dict(state_dict)
        return model

    def evaluate_accuracy(self):
        num_classes = len(self.dataloader.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        self.model.eval()
        data_iter = iter(self.dataloader.test_loader)

        for (xs, ys) in tqdm(data_iter):
            with torch.no_grad():
                ys = ys.to(self.device)
                xs = xs.to(self.device)
                ys_pred, _ = self.model.forward(xs)

            ys_pred = torch.argmax(ys_pred, dim=1)

            for y_pred, y_true in zip(ys_pred, ys):
                confusion_matrix[y_true][y_pred] += 1

        acc = acc_from_cm(confusion_matrix)

        return acc

    def find_k_nearest_patches_to_prototypes(
        self,
        dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
        k=3,
        root_dir_for_saving_images="./nearest",
    ):
        self.model.eval()
        n_prototypes = self.model.num_prototypes
        prototype_shape = self.model.prototype_shape
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
        protoL_rf_info = self.model.proto_layer_rf_info

        heaps = []
        for _ in range(n_prototypes):
            heaps.append([])

        for (search_batch, search_y) in tqdm(dataloader):

            with torch.no_grad():
                search_batch = search_batch.to(self.device)
                _, proto_dist_torch = self.model.push_forward(search_batch)

            proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

            for img_idx, distance_map in enumerate(proto_dist_):
                for j in range(n_prototypes):
                    closest_patch_distance_to_prototype_j = np.amin(distance_map[j])
                    closest_patch_indices_in_distance_map_j = list(
                        np.unravel_index(
                            np.argmin(distance_map[j], axis=None),
                            distance_map[j].shape,
                        )
                    )
                    closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
                    closest_patch_indices_in_img = compute_rf_prototype(
                        search_batch.size(2),
                        closest_patch_indices_in_distance_map_j,
                        protoL_rf_info,
                    )
                    closest_patch = search_batch[
                        img_idx,:,
                        closest_patch_indices_in_img[1] : closest_patch_indices_in_img[2],
                        closest_patch_indices_in_img[3] : closest_patch_indices_in_img[4],
                    ]
                    closest_patch = closest_patch.numpy()
                    closest_patch = np.transpose(closest_patch, (1, 2, 0))

                    original_img = search_batch[img_idx].numpy()
                    original_img = np.transpose(original_img, (1, 2, 0))

                    if self.model.prototype_activation_function == "log":
                        act_pattern = np.log(
                            (distance_map[j] + 1)
                            / (distance_map[j] + self.model.epsilon)
                        )
                    elif self.model.prototype_activation_function == "linear":
                        act_pattern = max_dist - distance_map[j]
                    else:
                        raise NotImplementedError()

                    patch_indices = closest_patch_indices_in_img[1:5]

                    closest_patch = ImagePatch(
                        patch=closest_patch,
                        label=search_y[img_idx],
                        distance=closest_patch_distance_to_prototype_j,
                        original_img=original_img,
                        act_pattern=act_pattern,
                        patch_indices=patch_indices,
                    )

                    if len(heaps[j]) < k:
                        heapq.heappush(heaps[j], closest_patch)
                    else:
                        heapq.heappushpop(heaps[j], closest_patch)


        for j in range(n_prototypes):
            heaps[j].sort()
            heaps[j] = heaps[j][::-1]
            dir_for_saving_images = os.path.join(root_dir_for_saving_images, "%05d" % j)
            makedir(dir_for_saving_images)
            labels = []

            for i, patch in enumerate(heaps[j]):

                plt.imsave(
                    fname=os.path.join(
                        dir_for_saving_images,
                        "%02d_" % (i + 1) + "nearest" + "_original.png",
                    ),
                    arr=patch.original_img,
                    vmin=0.0,
                    vmax=1.0,
                )

                img_size = patch.original_img.shape[0]
                upsampled_act_pattern = cv2.resize(
                    patch.act_pattern,
                    dsize=(img_size, img_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                rescaled_act_pattern = upsampled_act_pattern - np.amin(
                    upsampled_act_pattern
                )
                rescaled_act_pattern = rescaled_act_pattern / np.amax(
                    rescaled_act_pattern
                )
                heatmap = cv2.applyColorMap(
                    np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET
                )
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]
                overlayed_original_img = 0.5 * patch.original_img + 0.3 * heatmap

                # if different from original image, save the patch (i.e. receptive field)
                if patch.patch.shape[0] != img_size or patch.patch.shape[1] != img_size:
                    np.save(
                        os.path.join(
                            dir_for_saving_images,
                            "%02d_" % (i + 1)
                            + "nearest-"
                            + "_receptive_field_indices.npy",
                        ),
                        patch.patch_indices,
                    )
                    plt.imsave(
                        fname=os.path.join(
                            dir_for_saving_images,
                            "%02d_" % (i + 1) + "nearest-" + "_receptive_field.png",
                        ),
                        arr=patch.patch,
                        vmin=0.0,
                        vmax=1.0,
                    )
                    # save the receptive field patch with heatmap
                    overlayed_patch = overlayed_original_img[
                        patch.patch_indices[0] : patch.patch_indices[1],
                        patch.patch_indices[2] : patch.patch_indices[3],
                        :,
                    ]
                    plt.imsave(
                        fname=os.path.join(
                            dir_for_saving_images,
                            "%02d_" % (i + 1)
                            + "nearest-"
                            + "_receptive_field_with_heatmap.png",
                        ),
                        arr=overlayed_patch,
                        vmin=0.0,
                        vmax=1.0,
                    )

                # save the highly activated patch
                high_act_patch_indices = find_high_activation_crop(
                    upsampled_act_pattern
                )
                high_act_patch = patch.original_img[
                    high_act_patch_indices[0] : high_act_patch_indices[1],
                    high_act_patch_indices[2] : high_act_patch_indices[3],
                    :,
                ]

                plt.imsave(
                    fname=os.path.join(
                        dir_for_saving_images,
                        "%02d_" % (i + 1) + "nearest" + "_high_act_patch.png",
                    ),
                    arr=high_act_patch,
                    vmin=0.0,
                    vmax=1.0,
                )

                imsave_with_bbox(
                    fname=os.path.join(
                        dir_for_saving_images,
                        "%02d_" % (i + 1)
                        + "nearest"
                        + "_high_act_patch_in_original_img.png",
                    ),
                    img_rgb=patch.original_img,
                    bbox_height_start=high_act_patch_indices[0],
                    bbox_height_end=high_act_patch_indices[1],
                    bbox_width_start=high_act_patch_indices[2],
                    bbox_width_end=high_act_patch_indices[3],
                    color=(0, 255, 255),
                )

            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_images, "class_id.npy"), labels)

        labels_all_prototype = np.array(
            [[patch.label for patch in heaps[j]] for j in range(n_prototypes)]
        )

        np.save(
            os.path.join(root_dir_for_saving_images, "full_class_id.npy"),
            labels_all_prototype,
        )

        return labels_all_prototype
