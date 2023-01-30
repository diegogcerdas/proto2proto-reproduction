import torch, os, heapq, cv2
from .lib.protopnet.model import PPNet
from .lib.features import init_backbone
from .lib.utils.evaluate import acc_from_cm
from .lib.protopnet.receptive_field import compute_rf_prototype
from .lib.protopnet.find_nearest import imsave_with_bbox, ImagePatch
from .lib.protopnet.helpers import makedir, find_high_activation_crop
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .dataloader import CUBDataLoader


class PPNetWrapper:
    def __init__(self, args, device):
        self.args = args
        self.dataloader = CUBDataLoader(args)
        self.device = device
        self.model = self.load_model().to(device)
        self.indices_scores = None

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

    def compute_accuracy(self):
        num_classes = len(self.dataloader.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        self.model.eval()
        data_iter = iter(self.dataloader.test_loader_norm)

        for (xs, ys) in tqdm(data_iter, desc="Computing accuracy"):
            with torch.no_grad():
                ys = ys.to(self.device)
                xs = xs.to(self.device)
                ys_pred, _ = self.model.forward(xs)

            ys_pred = torch.argmax(ys_pred, dim=1)

            for y_pred, y_true in zip(ys_pred, ys):
                confusion_matrix[y_true][y_pred] += 1

        acc = acc_from_cm(confusion_matrix)

        return acc

    def compute_indices_scores(self):
        self.model.eval()
        data_iter = iter(self.dataloader.test_loader)
        data = torch.empty(0)

        for (xs, _) in tqdm(data_iter, desc="Computing indices and scores"):
            with torch.no_grad():
                xs = xs.to(self.device)
                distances, _, _ = self.model.prototype_distances(xs)
                scores, indices = F.max_pool2d(-distances,
                                                 kernel_size=(distances.size()[2],
                                                              distances.size()[3]),
                                                 return_indices=True)
                indices = indices.view(indices.shape[0], 1, self.model.num_prototypes)
                scores = scores.view(scores.shape[0], 1, self.model.num_prototypes)
                data = torch.cat([data, torch.cat([indices, scores], dim=1)], dim=0)

        self.indices_scores = data

    def compute_ajs(self, teacher_indices_scores, dist_th):
        assert self.indices_scores is not None, "Please run self.compute_indices_scores()"
        assert teacher_indices_scores is not None, "teacher_indices_scores cannot be None"

        num_test_images = len(self.dataloader.test_loader)
        iou_student = 0.0

        for ii in tqdm(range(num_test_images), desc="Computing AJS"):
            if dist_th is None:
                pruned_teacher_indices = teacher_indices_scores[ii, 0]
                pruned_student_indices = self.indices_scores[ii, 0]
            else:
                pruned_teacher_indices = []
                for jj, score in enumerate(teacher_indices_scores[ii, 1]):
                    if abs(-score) <= dist_th:
                        pruned_teacher_indices.append(teacher_indices_scores[ii, 0, jj])
                pruned_student_indices = []
                for jj, score in enumerate(self.indices_scores[ii, 1]):
                    if abs(-score) <= dist_th:
                        pruned_student_indices.append(self.indices_scores[ii, 0, jj])

            iou_student += jaccard_similarity_basic(pruned_teacher_indices, pruned_student_indices)

        ajs = iou_student / num_test_images

        return ajs

    def compute_aap(self, dist_th):
        assert self.indices_scores is not None, "Please run self.compute_indices_scores()"

        num_test_images = len(self.dataloader.test_loader)
        count_student = 0

        for ii in tqdm(range(num_test_images), desc="Computing AAP"):
            if dist_th is None:
                pruned_student_indices = self.indices_scores[ii, 0]
            else:
                pruned_student_indices = []
                for jj, score in enumerate(self.indices_scores[ii, 1]):
                    if abs(-score) <= dist_th:
                        pruned_student_indices.append(self.indices_scores[ii, 0, jj])
                        
            count_student += len(set(pruned_student_indices))

        aap = count_student / num_test_images

        return aap

    # def compute_pms(self):

    #     # Teacher, Student-kd IoU
    #     mm = self.teacher_model.module.num_prototypes
    #     nn = self.student_kd_model.module.num_prototypes

    #     tchr_proto_id = ray.put(teacher_prototypes)
    #     stu_kd_proto_id = ray.put(student_kd_prototypes)
    #     stu_baseline_proto_id = ray.put(student_baseline_prototypes)
    #     max_union_list = [ii for ii in range(int(0.1*num_test_images), num_test_images, int(0.1*num_test_images))]

    #     cost_kd_list = []
    #     for max_union in max_union_list:

    #         iou_matrix = np.zeros((mm,nn))
    #         obj_ids = []
    #         for ii in tqdm(range(mm)):

    #             obj_id = jaccard_row.remote(ii, tchr_proto_id, stu_kd_proto_id, max_union)
    #             obj_ids.append(obj_id)

    #             if ii % 30 == 0 or ii == mm - 1:
    #                 results = ray.get(obj_ids)
    #                 for kk in range(len(obj_ids)):
    #                     index, sim = results[kk]
    #                     iou_matrix[index] = sim
    #                 obj_ids = []

    #         assert len(obj_ids) == 0

    #         iou_distance_matrix = 1.0 - iou_matrix
    #         r_ts , c_stu_kd = linear_sum_assignment(iou_distance_matrix)
    #         cost_kd = iou_distance_matrix[r_ts, c_stu_kd].sum() / len(r_ts)
    #         cost_kd_list.append(cost_kd)

    #     avg_cost_kd = sum(cost_kd_list) / len(cost_kd_list)
    #     print("Average Similarity between prototypes(KD)", 1.0 - avg_cost_kd)

    #     return

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

        for (search_batch, search_y) in tqdm(dataloader, desc="Finding patches"):

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

def jaccard_similarity_basic(list1, list2):

    s1 = set(list1)
    s2 = set(list2)

    intersect = len(s1.intersection(s2))
    union = (len(s1) + len(s2)) - intersect

    if union == 0:
        return 0.0

    sim = float(intersect/union)

    return sim
