import torch, os, heapq, cv2
from .lib.protopnet.model import PPNet
from .lib.features import init_backbone
from .lib.utils.evaluate import acc_from_cm
from .lib.protopnet.receptive_field import compute_rf_prototype
from .lib.protopnet.find_nearest import imsave_with_bbox, ImagePatch
from .lib.protopnet.helpers import makedir, find_high_activation_crop
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .dataloader import CUBDataLoader


class PPNetWrapper:
    def __init__(self, args, device):
        self.args = args
        self.dataloader = CUBDataLoader(args)
        self.device = device
        self.model = self.load_model().to(device)
        self.indices_scores = None
        self.save_img_dir = args.saveImgDir

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

    def compute_indices_scores(self):
        self.model.eval()
        data_iter = iter(self.dataloader.test_loader)
        data = torch.empty(0)

        for (xs, _) in tqdm(data_iter, desc="Computing indices and scores"):
            with torch.no_grad():
                xs = xs.to(self.device)
                distances, _, _ = self.model.prototype_distances(xs)
                scores, indices = F.max_pool2d(
                    -distances,
                    kernel_size=(distances.size()[2], distances.size()[3]),
                    return_indices=True,
                )
                indices = indices.view(indices.shape[0], 1, self.model.num_prototypes)
                scores = scores.view(scores.shape[0], 1, self.model.num_prototypes)
                data = torch.cat([data, torch.cat([indices, scores], dim=1)], dim=0)

        self.indices_scores = data

    def compute_ajs(self, dist_th, teacher_indices_scores):
        assert (
            self.indices_scores is not None
        ), "Please run self.compute_indices_scores()"
        assert (
            teacher_indices_scores is not None
        ), "teacher_indices_scores cannot be None"

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

            iou_student += jaccard_similarity_basic(
                pruned_teacher_indices, pruned_student_indices
            )

        ajs = iou_student / num_test_images

        return ajs

    def compute_aap(self, dist_th):
        assert (
            self.indices_scores is not None
        ), "Please run self.compute_indices_scores()"

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

    def compute_pms(self, dist_th, teacher_indices_scores):
        assert (
            self.indices_scores is not None
        ), "Please run self.compute_indices_scores()"
        assert (
            teacher_indices_scores is not None
        ), "teacher_indices_scores cannot be None"

        num_test_images = len(self.dataloader.test_loader)
        num_prototypes = self.model.num_prototypes

        teacher_prototypes = [[[], []] for _ in range(num_prototypes)]
        student_prototypes = [[[], []] for _ in range(num_prototypes)]

        for ii in tqdm(range(num_test_images), desc="Computing PMS"):

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

            for jj in range(len(teacher_prototypes)):
                if jj <= len(pruned_teacher_indices) - 1:
                    name = "%04d" % ii + "%02d" % pruned_teacher_indices[jj]
                    teacher_prototypes[jj][0].append(name)
                    teacher_prototypes[jj][1].append(None)

            for jj in range(len(student_prototypes)):
                if jj <= len(pruned_student_indices) - 1:
                    name = "%04d" % ii + "%02d" % pruned_student_indices[jj]
                    student_prototypes[jj][0].append(name)
                    student_prototypes[jj][1].append(None)

        max_union_list = [
            ii
            for ii in range(
                int(0.1 * num_test_images), num_test_images, int(0.1 * num_test_images)
            )
        ]

        lowest_cost = np.inf
        best_allocation = None
        cost_list = []
        for max_union in max_union_list:
            iou_matrix = np.zeros((num_prototypes, num_prototypes))
            for ii, teacher_prototype in enumerate(tqdm(teacher_prototypes)):
                proto_row = jaccard_row(
                    teacher_prototype, student_prototypes, max_union
                )
                iou_matrix[ii] = proto_row
            iou_distance_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(iou_distance_matrix)
            cost = iou_distance_matrix[row_ind, col_ind].sum() / len(row_ind)
            if cost < lowest_cost:
                lowest_cost = cost
                best_allocation = (row_ind, col_ind)
            cost_list.append(cost)

        avg_cost = sum(cost_list) / len(cost_list)
        pms = 1.0 - avg_cost

        return pms, best_allocation

    def compute_accuracy(self):
        num_classes = len(self.dataloader.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        self.model.eval()
        data_iter = iter(self.dataloader.test_loader)

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

    def find_nearest_patches(self, k=5):
        self.model.eval()
        n_prototypes = self.model.num_prototypes
        prototype_shape = self.model.prototype_shape
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
        protoL_rf_info = self.model.proto_layer_rf_info

        heaps = []
        for _ in range(n_prototypes):
            heaps.append([])

        for (search_batch, search_y) in tqdm(
            self.dataloader.project_loader, desc="Finding patches"
        ):

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
                    closest_patch_indices_in_distance_map_j = [
                        0
                    ] + closest_patch_indices_in_distance_map_j
                    closest_patch_indices_in_img = compute_rf_prototype(
                        search_batch.size(2),
                        closest_patch_indices_in_distance_map_j,
                        protoL_rf_info,
                    )
                    closest_patch = search_batch[
                        img_idx,
                        :,
                        closest_patch_indices_in_img[1] : closest_patch_indices_in_img[
                            2
                        ],
                        closest_patch_indices_in_img[3] : closest_patch_indices_in_img[
                            4
                        ],
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

        for j in tqdm(range(n_prototypes), desc="Saving images"):
            heaps[j].sort()
            heaps[j] = heaps[j][::-1]
            dir_for_saving_images = os.path.join(self.save_img_dir, "%05d" % j)
            makedir(dir_for_saving_images)

            for i, patch in enumerate(heaps[j]):

                img_size = patch.original_img.shape[0]
                upsampled_act_pattern = cv2.resize(
                    patch.act_pattern,
                    dsize=(img_size, img_size),
                    interpolation=cv2.INTER_CUBIC,
                )

                high_act_patch_indices = find_high_activation_crop(
                    upsampled_act_pattern
                )

                imsave_with_bbox(
                    fname=os.path.join(
                        dir_for_saving_images,
                        "%02d" % (i + 1) + ".png",
                    ),
                    img_rgb=patch.original_img,
                    bbox_height_start=high_act_patch_indices[0],
                    bbox_height_end=high_act_patch_indices[1],
                    bbox_width_start=high_act_patch_indices[2],
                    bbox_width_end=high_act_patch_indices[3],
                    color=(0, 255, 255),
                )

        labels_all_prototype = np.array(
            [[patch.label for patch in heaps[j]] for j in range(n_prototypes)]
        )

        np.save(
            os.path.join(self.save_img_dir, "class_ids.npy"),
            labels_all_prototype,
        )

        return labels_all_prototype


def jaccard_row(teacher_prototype, student_prototypes, max_union):

    proto_row = np.zeros(len(student_prototypes))
    for jj in range(len(student_prototypes)):
        proto_row[jj] = jaccard_similarity(
            teacher_prototype, student_prototypes[jj], max_union=max_union
        )

    return proto_row


def jaccard_similarity(list1, list2, max_union=100000.0):

    s1 = set(list1[0])
    s2 = set(list2[0])

    intersect = len(s1.intersection(s2))
    union = (len(s1) + len(s2)) - intersect

    if union == 0:
        return 0.0
    elif intersect >= max_union:
        return 1.0
    else:
        sim = float(intersect / min(union, max_union))

    return sim


def jaccard_similarity_basic(list1, list2):

    s1 = set(list1)
    s2 = set(list2)

    intersect = len(s1.intersection(s2))
    union = (len(s1) + len(s2)) - intersect

    if union == 0:
        return 0.0

    sim = float(intersect / union)

    return sim
