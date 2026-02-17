from typing import Optional, Union, List
#import cvxpy as cp
import numpy as np
import torch
import scipy.stats
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, minimize, Reals, SolverFactory
from src.iwmv import *
from src.ds import *

    
class InterAnnotatorAgreementAPI:
    def __init__(self, annotations: np.array, optim_norm_p: int = 2, label_distribution: Optional[List[float]] = None,
                 tolerance: float = -1e-8):
        assert len(annotations.shape) == 2, "Annotations should have only two dimensions (dataset_size, num_annotators)"
        self._annotations = annotations
        self._optim_norm_p = optim_norm_p
        self._tolerance = tolerance
        self._compute_statistics()
        self._wrapper_compute_label_distribution(label_distribution)
        self.is_t_matrix_initialized = False  # We initialize the t-matrix only the first time that get_posterior_probability() is called.

    def _wrapper_compute_label_distribution(self, label_distribution: Optional[List[float]]) -> None:
        """
        :param label_distribution: it is a optional float list that sum up to 1, where the i-th element represents a probability for class i.
        :return: None
        """
        if label_distribution is None:
            label_distribution = self._compute_label_distribution(self._annotations)
        self._label_distribution = label_distribution
        assert (
                round(sum(self._label_distribution), 4) == 1
        ), "There is something wrong with your label distribution. It does not sum up to 1."

    def _to_one_hot(self, targets: np.array):
        """
        :param targets: it transforms a list of class
        :return: a numpy array with one-hot encoding
        """
        # target of the shame (dataset_size, num_annotators) to (dataset_size, num_annotators, num_classes)
        res = np.eye(self._num_classes)[targets.reshape(-1)]
        return res.reshape(list(targets.shape) + [self._num_classes])

    def _build_t_matrix(self, check_status: bool = True) -> None:
        assert not self.is_t_matrix_initialized, "T matrix is already optimized. You should construct a new object."
        self._d_matrix = self._D_estimator()
        self._m_matrix = self._M_estimator()

        self._optim_second_term = self._compute_optim_second_term(self._d_matrix, self._m_matrix)
        #self._t_hat = self._optimize_T(self._optim_second_term, self._optim_norm_p, check_status)
        self._t_hat = self.solve_problem(self._optim_second_term, self._optim_norm_p, check_status)
        assert (self._t_hat >= self._tolerance).all(), self._t_hat[self._t_hat < self._tolerance]
        #assert np.equal(self._t_hat, self._t_hat.T).all(), f"Error: T is not symmetric. T matrix is:\n {self._t_hat.T}"
        self._fix_negative_values_t_matrix()
        self.is_t_matrix_initialized = True

    def _fix_negative_values_t_matrix(self):
        if not (self._t_hat >= 0).all():
            print("WARNING: WE MANUALLY REMOVED NEGATIVE NUMBERS FROM THE T MATRIX")
            self._t_hat[self._t_hat < 0] = 0

    def __repr__(self) -> str:
        return f"Num annotators: {self._num_annotators}, Dataset Size: {self._num_samples}, Num Classes: {self._num_classes}, Classes: {self._classes}, Class distribution: {self._label_distribution}"

    def _compute_statistics(self) -> None:
        self._num_samples, self._num_annotators = self._annotations.shape
        self._classes = sorted(list(set(np.hstack(self._annotations))))  # set of all the possible classes
        self._num_classes = len(self._classes)
        assert max(self._classes) == self._num_classes - 1, "Your dataset does not contains all the classes"

    def _compute_label_distribution(self, noisy_y: np.array) -> List[float]:
        return [np.mean(noisy_y == c) for c in self._classes]

    def _D_estimator(self) -> np.array:
        """
        :return: a matrix of shape (num_classes, num_classes) with the label distribution in the diagonal.
        """
        return np.diag(self._label_distribution)

    def _M_estimator(self) -> np.array:
        hat_m = np.zeros((self.num_classes, self.num_classes))
        for ann_a in range(self.num_annotators):
            for ann_b in range(ann_a + 1, self.num_annotators):
                for i in range(self.num_samples):
                    class_a = self.annotations[i, ann_a]
                    class_b = self.annotations[i, ann_b]
                    hat_m[class_a, class_b] += 1
                    hat_m[class_b, class_a] += 1
        hat_m /= (self.num_annotators * (self.num_annotators - 1) * self.num_samples)
        return hat_m


    def _compute_optim_second_term(self, D: np.array, M_hat: np.array) -> np.array:
        assert D.shape == (self._num_classes, self._num_classes) and M_hat.shape == (
            self._num_classes,
            self._num_classes,
        )
        app = np.linalg.inv(D ** 0.5)
        app = np.dot(app, np.dot(M_hat, app))
        app, U = np.linalg.eig(app)
        Λ = np.diag(app).astype(complex)  #complex needed to avoid NAN
        inv_U = U.T
        optim_second_term = np.dot(U, np.dot(Λ ** 0.5, inv_U))
        return optim_second_term



    def solve_problem(self, optim_second_term: np.array, norm_p: int, check_status: bool = True):
        model = ConcreteModel()

        # Create the decision variables
        model.t_hat = Var(range(self._num_classes), range(self._num_classes), domain=Reals, bounds=(0.00001, None))

        # Symmetric constraint
        def symmetric_constraint(model, i, j):
            return model.t_hat[i, j] == model.t_hat[j, i]

        model.symmetric_constraint = Constraint(range(self._num_classes), range(self._num_classes), rule=symmetric_constraint)

        # Add constraints
        def row_sum_constraint(model, i):
            return sum(model.t_hat[i, j] for j in range(self._num_classes)) == 1

        model.row_sum_constraint = Constraint(range(self._num_classes), rule=row_sum_constraint)

        def diagonal_constraint(model, i, j):
            if i == j:
                return model.t_hat[i, j] >= 0.5
            else:
                return Constraint.Skip

        model.diagonal_constraint = Constraint(range(self._num_classes), range(self._num_classes), rule=diagonal_constraint)

        # Define the objective function
        def objective_rule(model):
            return sum((model.t_hat[i, j] - optim_second_term[i, j].real) ** norm_p for i in range(self._num_classes) for j in range(self._num_classes))

        model.obj = Objective(rule=objective_rule, sense=minimize)

        # Solve the problem
        solver = SolverFactory('gurobi')
        solver.solve(model)

        # Check the solution status
        if model.obj() is  None:
            print("No optimal solution found.")
        optimal_matrix = np.zeros((self._num_classes, self._num_classes))
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                optimal_matrix[i, j] = model.t_hat[i, j].value
        return optimal_matrix
        
    
    def _optimize_T(self, optim_second_term: np.array, norm_p: int, check_status: bool = True):
        """
        This function find the optimal T in the space of the symmetric,
        stochastic matrices with elements on the diagonal > 0.5.
        We don't need T, I put it just as a control
        """
        # useful functions https://www.cvxpy.org/tutorial/functions/index.html
        t_hat = cp.Variable((self._num_classes, self._num_classes), symmetric=True)
        # Create two constraints.
        constraints = [
            t_hat[self._classes, self._classes] >= 0.5,
            cp.sum(t_hat, axis=1) == 1,
            t_hat >= 0.00001,  # 1e-5 approximation tollerance is 1e-8
        ]
        # I have to put 0.5 + eps because strict inequalities are not allowed
        # Form objective.
        objective = cp.Minimize(cp.norm(t_hat - optim_second_term, norm_p))
        # Form and solve problem.
        problem = cp.Problem(objective, constraints)
        problem.solve()  # Returns the optimal value.
        if problem.status != 'optimal':
            print(problem.status)    
        if check_status:
            assert problem.status == "optimal", "Error: Your T can't be optimized!"
        return t_hat.value

    def get_average_soft_labels(self, return_torch: bool = True) -> Union[np.array, torch.Tensor]:
        # the function does not require the T computation.
        one_hot = self._to_one_hot(self._annotations)
        # self._annotations (dataset_size, num_annotators) ---1-hot---> (dataset_size, num_annotators, num_classes)
        result = np.mean(one_hot, axis=1)
        return result if not return_torch else torch.from_numpy(result)

    def get_posterior_probability(self, return_torch: bool = True, check_status: bool = True) -> Union[np.array, torch.Tensor]:
        '''
        return a probability distribution as labels with the shape of (number of sample, num of annotators)
        '''
        if not self.is_t_matrix_initialized:
            # We initialize the t matrix only the first time that get_posterior_probability() is called.
            self._build_t_matrix(check_status)

        p_c = self._label_distribution * np.prod(self._t_hat.T[self._annotations], axis=1)
        result = p_c / np.sum(p_c, axis=-1)[:, None]
        assert (result >= 0).all(), "Error: get_posterior_probability() there are negative soft labels."
        return result if not return_torch else torch.from_numpy(result)
    

    def get_random_labels_from_annotations(self, return_torch: bool = True) -> Union[np.array, torch.Tensor]:
        """
        For each sample randomly select one of the labels chosen by the annotators.
        """
        indices = np.random.randint(self._num_annotators, size=self._num_samples)
        random_labels = self._annotations[np.arange(self._num_samples), indices]
        return random_labels if not return_torch else torch.from_numpy(random_labels)

    def get_iwmv_labels_from_annotations(self, return_torch: bool = True) -> Union[np.array, torch.Tensor]:
        """
        For each sample randomly select one of the labels chosen by the annotators.
        """
        e2wl, w2el, label_set = gete2wlandw2el(self._annotations)
        annotations, _, t_hat_iwmv = iwmv(e2wl, w2el, label_set, T_required=True)
        self._t_hat = t_hat_iwmv
        labels = dict2list(annotations)
        return labels if not return_torch else torch.from_numpy(labels)
    
    def get_dawid_skene_labels_from_annotations(self, return_torch: bool = True) -> Union[np.array, torch.Tensor]:
        """
        For each sample randomly select one of the labels chosen by the annotators.
        """
        e2wl, w2el, label_set = gete2wlandw2el(self._annotations)
        ds = EM(e2wl, w2el, label_set)
        annotations, _, T_matrix = ds.Run(T_required=True)
        self._t_hat = T_matrix
        annotations = ds.from_z_to_truths_multi(annotations,label_set)
        labels = dict2list(annotations)
        return labels if not return_torch else torch.from_numpy(labels)

    def get_random_labels(self, return_torch: bool = True) -> Union[np.array, torch.Tensor]:
        """
        For each sample randomly select one of the classes.
        It is completely independent by the annotations
        """
        result = np.random.choice(self._classes, size=self._num_samples)
        return result if not return_torch else torch.from_numpy(result)

    def get_hard_voting_annotations(self, return_torch: bool = True) -> Union[np.array, torch.Tensor]:
        """
        For each sample select the most common label in the annotations.
        In case of TIE it select the label with the minimum ID, as implemented in scipy.stats.mode.
        Hence, in that case [1, 1, 2, 2, 3, 4] --most common labels--> [1, 2] -> 1
        """
        result = np.array(scipy.stats.mode(self._annotations, axis=1).mode)
        return result if not return_torch else torch.from_numpy(result)

    def get_mixed(self, return_torch: bool = True) -> Union[np.array, torch.Tensor]:

        soft_labels = self.get_average_soft_labels(return_torch)
        posterior = self.get_posterior_probability(return_torch)
        if return_torch:
            result = torch.mean(torch.vstack((soft_labels.unsqueeze(0),posterior.unsqueeze(0))), dim=0)
        else:
            result =  np.mean(np.stack((soft_labels, posterior), axis=2), axis=2)
        return result

    def return_annotations(self,aggregation_type:str='posterior', return_torch:bool=True) -> Union[np.array, torch.Tensor]:
        if aggregation_type == "posterior":
            labels = self.get_posterior_probability(return_torch=return_torch)
        elif aggregation_type == "majority":
            labels = self.get_hard_voting_annotations(return_torch=return_torch)
        elif aggregation_type == "random":
            labels = self.get_random_labels_from_annotations(return_torch=return_torch)
        elif aggregation_type == "average":
            labels = self.get_average_soft_labels(return_torch=return_torch)
        elif aggregation_type == "mixed":
            labels = self.get_mixed(return_torch=return_torch)
        elif aggregation_type == "iwmv":
            labels = self.get_iwmv_labels_from_annotations(return_torch=return_torch)
        elif aggregation_type == "ds":
            labels = self.get_dawid_skene_labels_from_annotations(return_torch=return_torch)
        else:
            raise ValueError("Aggregation type not recognized")
        return labels

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def classes(self) -> List[int]:
        return self._classes

    @property
    def annotations(self) -> np.array:
        return self._annotations

    @property
    def tolerance(self) -> int:
        return self.tolerance

    @property
    def num_annotators(self) -> int:
        return self._num_annotators

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def t_hat(self):
        if not self.is_t_matrix_initialized:
            return None
        return self._t_hat

    @property
    def label_distribution(self) -> List[float]:
        return self._label_distribution