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
from .arguments import Arguments
from sklearn import preprocessing
from typing import List, Tuple


class PPNetWrapper:
    """Wrapper class for evaluation of ProtoPNet architectures
    in the context of Proto2Proto.
    """

    def __init__(self, args: Arguments, device: str):
        self.args = args
        self.dataloader = CUBDataLoader(args)
        self.device = device
        self.model = self.load_model().to(device)
        self.indices_scores = None
        self.save_img_dir = args.saveImgDir

    def load_model(self) -> PPNet:
        """Initializes a ProtoPNet model
        from the checkpoint specified in the arguments.

        Returns:
            PPNet: initialized model.
        """
        features, _ = init_backbone(self.args.backbone)
        model = PPNet(
            num_classes=len(self.dataloader.classes),
            feature_net=features,
            args=self.args,
        )
        state_dict = torch.load(self.args.backbone.loadPath)
        model.load_state_dict(state_dict)
        return model

    def compute_indices_scores(self) -> None:
        """Computes a list with a tuple for each training example:
        - First element lists an input patch index for each of the model's prototypes
        - Second element lists the similarity score between the corresponding input patch and prototype
        """
        self.model.eval()
        data_iter = iter(self.dataloader.test_loader)
        data = []

        for (xs, _) in tqdm(data_iter, desc="Computing indices and scores"):
            with torch.no_grad():
                xs = xs.to(self.device)
                distances, _, _ = self.model.prototype_distances(xs)
                scores, indices = F.max_pool2d(
                    -distances,
                    kernel_size=(distances.size()[2], distances.size()[3]),
                    return_indices=True,
                )

                indices = (
                    indices.view(self.model.num_prototypes)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                scores = (
                    scores.view(self.model.num_prototypes)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                data.append([indices, scores])

        self.indices_scores = data

    def compute_ajs(self, dist_th: float, teacher_indices_scores: List[Tuple]) -> float:
        """Computes Average Jaccard Similarity
        between the current model and a teacher model

        Args:
            dist_th (float): Defines the threshold for defining an active patch
            teacher_indices_scores (List[Tuple]): Indices and scores of the teacher model

        Returns:
            float: AJS score
        """
        assert (
            self.indices_scores is not None
        ), "Please run self.compute_indices_scores()"
        assert (
            teacher_indices_scores is not None
        ), "teacher_indices_scores cannot be None"

        num_test_images = len(self.dataloader.test_loader)
        iou_student = 0.0

        # Loop through test images
        for ii in tqdm(range(num_test_images), desc="Computing AJS"):
            # Define active patches
            if dist_th is None:
                pruned_teacher_indices = teacher_indices_scores[ii][0]
                pruned_student_indices = self.indices_scores[ii][0]
            else:
                pruned_teacher_indices = []
                for jj, score in enumerate(teacher_indices_scores[ii][1]):
                    if abs(-score) <= dist_th:
                        pruned_teacher_indices.append(teacher_indices_scores[ii][0][jj])
                pruned_student_indices = []
                for jj, score in enumerate(self.indices_scores[ii][1]):
                    if abs(-score) <= dist_th:
                        pruned_student_indices.append(self.indices_scores[ii][0][jj])

            # Keep sum of Jaccard similarities
            iou_student += jaccard_similarity_basic(
                pruned_teacher_indices, pruned_student_indices
            )

        # Compute average Jaccard similarity
        ajs = float(iou_student / num_test_images)

        return ajs

    def compute_aap(self, dist_th: float) -> float:
        """Compute Average Active Patches for the current model

        Args:
            dist_th (float): Defines the threshold for defining an active patch

        Returns:
            float: AAP score
        """
        assert (
            self.indices_scores is not None
        ), "Please run self.compute_indices_scores()"

        num_test_images = len(self.dataloader.test_loader)
        count_student = 0

        # Loop through test images
        for ii in tqdm(range(num_test_images), desc="Computing AAP"):
            if dist_th is None:
                pruned_student_indices = self.indices_scores[ii][0]
            else:
                pruned_student_indices = []
                for jj, score in enumerate(self.indices_scores[ii][1]):
                    if abs(-score) <= dist_th:
                        pruned_student_indices.append(self.indices_scores[ii][0][jj])

            # Sum number of active patches
            count_student += len(set(pruned_student_indices))

        # Compute average active patches
        aap = float(count_student / num_test_images)

        return aap

    def compute_pms(self, teacher_indices_scores: List[Tuple]) -> Tuple[float, Tuple]:
        """Computes Pattern-Matching Score

        Args:
            teacher_indices_scores (List[Tuple]): Indices and scores of the teacher model

        Returns:
            Tuple[float, Tuple]: PMS score and 
                a tuple (row_indices, column_indices) matching student and teacher prototypes
        """
        assert (
            self.indices_scores is not None
        ), "Please run self.compute_indices_scores()"
        assert (
            teacher_indices_scores is not None
        ), "teacher_indices_scores cannot be None"

        num_test_images = len(self.dataloader.test_loader)
        num_prototypes = self.model.num_prototypes

        teacher_prototypes = np.empty((num_prototypes, num_test_images))
        student_prototypes = np.empty((num_prototypes, num_test_images))

        # Loop through test images
        for ii in tqdm(range(num_test_images), desc="Computing PMS"):

            # We consider all patches as active (as in original Proto2Proto implementation)
            teacher_indices = teacher_indices_scores[ii][0]
            student_indices = self.indices_scores[ii][0]

            # Create an ID for each active patch in the image (Teacher network)
            for jj in range(len(teacher_prototypes)):
                name = "%04d" % ii + "_" + "%02d" % teacher_indices[jj]
                teacher_prototypes[jj][ii] = name

            # Create an ID for each active patch in the image (Student network)
            for jj in range(len(student_prototypes)):
                name = "%04d" % ii + "_" + "%02d" % student_indices[jj]
                student_prototypes[jj][ii] = name

        # Following code is introduced for efficiency: uses Tensors and precomputes sets
        le = preprocessing.LabelEncoder()
        all_prototypes = np.concatenate(
            [teacher_prototypes, student_prototypes], axis=0
        )
        all_shape = all_prototypes.shape
        teacher_size = teacher_prototypes.shape[0]
        all_prototypes = np.reshape(all_prototypes, (-1))
        encoded_names = le.fit_transform(all_prototypes)
        encoded_names = torch.as_tensor(encoded_names).view(all_shape).to(self.device)
        teacher_prototypes = encoded_names[:teacher_size]
        student_prototypes = encoded_names[teacher_size:]

        teacher_prototypes = [set(p.unique().cpu().numpy()) for p in teacher_prototypes]
        student_prototypes = [set(p.unique().cpu().numpy()) for p in student_prototypes]

        # Define max_union hyperparameters (from alpha value in Proto2Proto paper)
        max_union_list = [
            ii
            for ii in range(
                int(0.1 * num_test_images), num_test_images, int(0.1 * num_test_images)
            )
        ]

        lowest_cost = np.inf
        best_allocation = None
        cost_list = []
        # For each max_union hyperparameter...
        for max_union in max_union_list:
            iou_matrix = torch.zeros((num_prototypes, num_prototypes)).to(self.device)
            # For each teacher prototype...
            for ii, teacher_prototype in enumerate(tqdm(teacher_prototypes)):
                # Compute Jaccard similarity with each student prototypes
                proto_row = jaccard_row(
                    teacher_prototype, student_prototypes, max_union, self.device
                )
                iou_matrix[ii] = proto_row
            # Translate similarity to distances
            iou_distance_matrix = (1.0 - iou_matrix).cpu().numpy()
            # Compute teacher-student prototype matching through Hungarian Matching algorithm
            row_ind, col_ind = linear_sum_assignment(iou_distance_matrix)
            # Compute overall cost of allocation
            cost = iou_distance_matrix[row_ind, col_ind].sum() / len(row_ind)
            # Keep track of allocation with lowest cost
            if cost < lowest_cost:
                lowest_cost = cost
                best_allocation = (row_ind, col_ind)
            cost_list.append(cost)

        # Average costs across hyperparameters
        avg_cost = float(sum(cost_list) / len(cost_list))
        pms = 1.0 - avg_cost  # Translate to PMS score

        return pms, best_allocation

    def compute_accuracy(self) -> float:
        """Computes classification accuracy on test set

        Returns:
            float: obtained accuracy
        """
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

        acc = float(acc_from_cm(confusion_matrix))

        return acc

    def find_nearest_patches(self, k: int = 5) -> None:
        """Finds and stores nearest training image patches for each of the model's prototypes
        Stores the original image pointing indicating the found patch in the folder
        specified by self.save_img_dir

        Args:
            k (int, optional): Find top-k activated patches. Defaults to 5.
        """
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
                    closest_patch = closest_patch.cpu().numpy()
                    closest_patch = np.transpose(closest_patch, (1, 2, 0))

                    original_img = search_batch[img_idx].cpu().numpy()
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


def jaccard_row(teacher_prototype, student_prototypes, max_union, device):
    """Computes Jaccard similarity between a teacher prototype's set of active patches
    and a list with student prototypes' sets of active patches.
    """

    proto_row = torch.zeros(len(student_prototypes)).to(device)
    for jj in range(len(student_prototypes)):
        proto_row[jj] = jaccard_similarity(
            teacher_prototype, student_prototypes[jj], max_union=max_union
        )

    return proto_row


def jaccard_similarity(s1, s2, max_union=100000.0):
    """
    Computing Jaccard similarity between two sets.
    Aids computation of Modified Jaccard similarity,
    as defined in Proto2Proto
    """

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
    """
    Simple method for computing Jaccard similarity between two lists.
    """

    s1 = set(list1)
    s2 = set(list2)

    intersect = len(s1.intersection(s2))
    union = (len(s1) + len(s2)) - intersect

    if union == 0:
        return 0.0

    sim = float(intersect / union)

    return sim
