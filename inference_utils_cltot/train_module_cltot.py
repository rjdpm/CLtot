from typing import Tuple, Dict, List, Any
import os, sys, pickle, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import r2_score

import torch, platform
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils_cltot import regression_test_metrics, datetime_now, create_folder, subset_loader, get_func_input_names

class EarlyStopping:
    def __init__(self, stop_patience: int = 10, min_delta: float = 1e-5, type_: str = 'loss'):
        """
        Early stopping mechanism for model training.

        Args:
            stop_patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as improvement.
            type_ (str): Metric type to monitor ('loss' or 'accuracy').

        Raises:
            ValueError: If `type_` is not 'loss' or 'accuracy'.
        """
        if type_ not in ['loss', 'accuracy']:
            raise ValueError("type_ must be either 'loss' or 'accuracy'")

        self.patience = stop_patience
        self.min_delta = min_delta
        self.type_ = type_
        self.counter = 0
        self.early_stop = False
        self.best_score = np.inf if self.type_ == 'loss' else -np.inf

    def __call__(self, current_score: float):
        """
        Update the early stopping status based on the current validation metric.

        Args:
            current_score (float): Current validation loss or accuracy.

        Returns:
            None: Updates internal counter and early_stop flag.
        """
        if self.type_ == 'loss':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        elif self.type_ == 'accuracy':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    
                
class Trainer_functions:
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_scheduler(scheduler_name: str,
                    optimizer: torch.optim.Optimizer,
                    **kwargs
                    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Returns a learning rate scheduler based on the provided name and parameters.
        
        Parameters:
            scheduler_name (str): Name of the scheduler, e.g., 'StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau'.
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.
            **kwargs: Additional keyword arguments for the specific scheduler.
            
        Returns:
            scheduler: The instantiated learning rate scheduler.
        """
        
        scheduler_name = scheduler_name.lower() 

        if scheduler_name == 'steplr':
            scheduler = StepLR(optimizer, **kwargs)
        elif scheduler_name == 'multisteplr':
            scheduler = MultiStepLR(optimizer, **kwargs)
        elif scheduler_name == 'exponentiallr':
            scheduler = ExponentialLR(optimizer, **kwargs)
        elif scheduler_name == 'reducelronplateau':
            scheduler = ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_name == 'cycliclr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, **kwargs)
        elif scheduler_name == 'onecyclelr':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Scheduler Name: '{scheduler_name}' is not in: ['steplr', 'multisteplr', 'exponentiallr', 'reducelronplateau', 'cycliclr', 'onecyclelr'].")

        return scheduler

    @staticmethod
    def activation_func(name='relu', param={'alpha':1.0, 'negative_slope':1e-2}):
        """
        Returns an activation function module based on the provided name.

        Args:
            name (str): Name of the activation function. Supported: 
                        'sigmoid', 'tanh', 'relu', 'selu', 'elu', 'leakyrelu', 'celu'.
            param (dict): Parameters for activation functions that require alpha or negative_slope.
                        Example: {'alpha': 1.0, 'negative_slope': 1e-2}

        Returns:
            nn.Module: Corresponding PyTorch activation function module.
        
        Raises:
            ValueError: If the activation name is not supported.
        """
            
        if name=='sigmoid':
            return nn.Sigmoid()
        elif name=='tanh':
            return nn.Tanh()
        elif name=='relu':
            return nn.ReLU(inplace=True)
        elif name=='selu':
            return nn.SELU(inplace=True)
        elif name=='elu':
            return nn.ELU(param['alpha'], inplace=True)
        elif name=='leakyrelu':
            return nn.LeakyReLU(param['negative_slope'], inplace=True)
        elif name=='celu':
            return nn.CELU(param['alpha'], inplace=True)
        else:
            raise ValueError('Define your activation function: "{}"' .format(name))


    @staticmethod
    def get_optimizer(model, solver_type='adam', learning_rate=1e-4, **kwargs):
        """
        Returns a PyTorch optimizer based on the specified type.

        Args:
            model (nn.Module): The neural network whose parameters are to be optimized.
            solver_type (str): Type of optimizer. Supported:
                'adam', 'sgd', 'rprop', 'rmsprop', 'radam', 'adamw', 'adagrad', 'adamax', 'nadam'
            learning_rate (float): Learning rate.
            kwargs (dict): Additional optimizer-specific parameters (e.g., weight_decay).

        Returns:
            torch.optim.Optimizer: Instantiated optimizer.

        Raises:
            ValueError: If an unsupported optimizer is specified.
        """
        solver_type = solver_type.lower()

        optimizers = {
            'adam':     optim.Adam,
            'sgd':      optim.SGD,
            'rprop':    optim.Rprop,
            'rmsprop':  optim.RMSprop,
            'radam':    getattr(optim, 'RAdam', None), 
            'adamw':    optim.AdamW,
            'adagrad':  optim.Adagrad,
            'adamax':   optim.Adamax,
            'nadam':    getattr(optim, 'NAdam', None), 
        }

        optimizer_class = optimizers.get(solver_type)

        if optimizer_class is None:
            raise ValueError(f'Unsupported optimizer: "{solver_type}". Please define or check availability.')

        return optimizer_class(model.parameters(), lr=learning_rate, **kwargs)

    #-------------------------------------------------------------------------------------------------------------------------
            

    @staticmethod
    def loss_function_selector(name='mse', params=None):
        
        """
        Returns a PyTorch loss function object based on the specified name.
        
        Args:
            name (str): Name of the loss function. Supported:
                'mse', 'l1', 'crossentropy', 'bce', 'bcewithlogits', 'smoothl1', 'huber',
                'nll', 'hinge', 'cosine', 'kl', 'marginranking', 'multilabelmargin'
            params (dict, optional): Parameters to pass into the loss function constructor.
        
        Returns:
            nn.Module: Instantiated loss function object.
        
        Raises:
            ValueError: If the loss name is unrecognized.
        """
        
        if params is None:
            params = {}

        name = name.lower()

        if name == 'mse':
            return nn.MSELoss(**params)
        elif name == 'l1':
            return nn.L1Loss(**params)
        elif name == 'crossentropy':
            return nn.CrossEntropyLoss(**params)
        elif name == 'bce':
            return nn.BCELoss(**params)
        elif name == 'bcewithlogits':
            return nn.BCEWithLogitsLoss(**params)
        elif name in ('smoothl1', 'huber'): 
            return nn.SmoothL1Loss(**params)
        elif name == 'nll':
            return nn.NLLLoss(**params)
        elif name == 'hinge':
            return nn.HingeEmbeddingLoss(**params)
        elif name == 'cosine':
            return nn.CosineEmbeddingLoss(**params)
        elif name == 'kl':
            return nn.KLDivLoss(**params)
        elif name == 'marginranking':
            return nn.MarginRankingLoss(**params)
        elif name == 'multilabelmargin':
            return nn.MultiLabelMarginLoss(**params)
        else:
            raise ValueError(f'Unknown loss function: "{name}". Please define it or add support.')


    @staticmethod
    def nn_input2output(model: nn.Module,
                        test_loader: DataLoader[Dict[str, torch.Tensor]],
                        criterion: nn.Module,
                        targets_col: str = 'labels',
                        description: str = 'Map',
                        device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        verbose:bool=False
                        ) -> Tuple[np.ndarray[float], np.ndarray[float], float]:
        
        """
        Evaluates the model on the test set and returns predictions, true values, and average loss.

        Args:
            model (nn.Module): Trained neural network model.
            test_loader (DataLoader): DataLoader containing test data batches.
            criterion (nn.Module): Loss function used for evaluation.
            targets_col (str): Key in each batch corresponding to target values.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            Tuple[np.ndarray, np.ndarray, float]: 
                - y_pred: Model predictions (NumPy array).
                - y_true: Ground-truth values (NumPy array).
                - epoch_test_loss: Average test loss over all samples.
        """
        
        model = model.to(device)
        model.eval() 
        y_pred = []
        y_true = []
        epoch_test_loss = 0
        model_params = get_func_input_names(model.forward)
        
        progress_bar=test_loader
        if verbose:
            progress_bar = tqdm(test_loader, desc=description)
            
        with torch.no_grad(): 
            for batch in progress_bar:
                targets = batch.pop(targets_col).to(device)
                try:
                    batch = {k: batch[k].to(device) for k in model_params if (k in batch.keys()) and (batch[k] is not None)}
                except:
                    batch = {k: [temp_v.to(device) for temp_v in batch[k] if temp_v is not None] for k in model_params if k in batch.keys()}
                
                outputs = model(**batch)
                loss = criterion(input=outputs.reshape(targets.shape).type(torch.float32), target=targets.type(torch.float32)).type(torch.float32)
                    
                # Collect all predictions and targets
                y_pred.append(outputs.reshape(targets.shape).cpu())
                y_true.append(targets.cpu())
                
                # Accumulate loss
                epoch_test_loss += loss.item()
            epoch_test_loss = epoch_test_loss/len(test_loader.dataset)
        
        # Concatenate all predictions and targets across batches
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
        
        return y_pred, y_true, epoch_test_loss

    # # Define the test function
    @staticmethod
    def test_model(model: nn.Module,
                test_loader: DataLoader[Dict[str, torch.Tensor]],
                criterion: nn.Module,
                targets_col: str = 'labels',
                description: str = 'Test',
                device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                verbose=True
                ) -> Tuple[Dict[str, float], float]:
        
        """
        Evaluates a trained model on the test data and computes regression metrics.

        Args:
            model (nn.Module): Trained neural network model.
            test_loader (DataLoader): DataLoader containing test data batches.
            criterion (nn.Module): Loss function used for evaluation.
            targets_col (str): Key in each batch corresponding to target values.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            Tuple[Dict[str, float], float]:
                - results: Dictionary containing regression metrics (e.g., R2, MAE, RMSE).
                - epoch_test_loss: Average test loss.
        """
        
        y_pred, y_true, epoch_test_loss = Trainer_functions.nn_input2output(model=model,
                                                                test_loader=test_loader,
                                                                criterion=criterion,
                                                                targets_col=targets_col,
                                                                description = description,
                                                                device=device,
                                                                verbose=verbose
                                                                )
        results = regression_test_metrics(y_true=y_true, y_pred=y_pred)
            
        return results, epoch_test_loss


    # # Define the test function
    @staticmethod
    def train_test_model(model: nn.Module,
                test_loader: DataLoader[Dict[str, torch.Tensor]],
                criterion: nn.Module,
                targets_col: str = 'labels',
                description: str = 'Train-Test',
                device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                verbose=True
                ) -> Tuple[float, float]:
        
        """
        Computes test R² score and average loss for a trained model.

        Args:
            model (nn.Module): Trained neural network model.
            test_loader (DataLoader): DataLoader containing test data batches.
            criterion (nn.Module): Loss function used for evaluation.
            targets_col (str): Key in each batch corresponding to target values.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            Tuple[float, float]:
                - r2: Coefficient of determination (R² score) on the test set.
                - epoch_test_loss: Average loss over the test set.
        """
        y_pred, y_true, epoch_test_loss = Trainer_functions.nn_input2output(model=model,
                                                                test_loader=test_loader,
                                                                criterion=criterion,
                                                                targets_col=targets_col,
                                                                description=description,
                                                                device=device,
                                                                verbose=verbose
                                                                )

        # Calculate metrics
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        
        return r2, epoch_test_loss


    @staticmethod
    def train_1epoch(model: nn.Module,
                    train_loader: DataLoader[Dict[str, torch.Tensor]],
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    param_reg: str = '',
                    lambda_reg: float = 0.1,
                    alpha: float = 0.5,
                    targets_col: str = 'labels',
                    description: str = 'Training',
                    device: str=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    verbose:bool = False
                    ) -> float:
            
        """
        Trains the model for one epoch using optional regularization and returns average training loss.

        Args:
            model (nn.Module): Model to be trained.
            train_loader (DataLoader): DataLoader containing training data batches.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
            param_reg (str): Type of regularization ('l1', 'l2', 'elastic_net', or '').
            lambda_reg (float): Regularization strength.
            alpha (float): Elastic net mixing parameter (0 = L2 only, 1 = L1 only).
            targets_col (str): Key in batch dict for ground truth labels.
            device (str): Device to perform computation on ('cuda' or 'cpu').

        Returns:
            float: Average training loss over the entire epoch.
        """
        
        model.train()
        model = model.to(device)
        epoch_train_loss = 0
        model_params = get_func_input_names(model.forward)
        
        progress_bar = enumerate(train_loader)
        if verbose:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=description)
        
        for i, batch in progress_bar:
            targets = batch.pop(targets_col).to(device)
            try:
                batch = {k: batch[k].to(device) for k in model_params if (k in batch.keys()) and (batch[k] is not None)}
            except:
                batch = {k: [temp_v.to(device) for temp_v in batch[k] if temp_v is not None] for k in model_params if k in batch.keys()}
            optimizer.zero_grad()
            outputs = model(**batch)
                
            loss = criterion(input=outputs.reshape(targets.shape).type(torch.float32), target=targets.type(torch.float32)).type(torch.float32)
            if param_reg in ['l1', 'elastic_net']:
                l1_temp = 0
                for param in model.parameters():
                    l1_temp = l1_temp + torch.sum(torch.abs(param))
                l1_reg = lambda_reg * l1_temp  # L1 term
                if param_reg == 'l1':
                    loss = loss + l1_reg
                
            if param_reg in ['l2', 'elastic_net']:
                l2_temp = 0
                for param in model.parameters():
                    l2_temp = l2_temp + torch.sum(param**2)
                l2_reg = lambda_reg * l2_temp    # L2 term
                if param_reg == 'l2':
                    loss = loss + l2_reg
                
            if param_reg == 'elastic_net':
                elastic_net_reg = alpha * l1_reg + (1 - alpha) * l2_reg
                loss = loss + elastic_net_reg
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.item()
        epoch_train_loss = epoch_train_loss/len(train_loader.dataset)
        
        return epoch_train_loss


    @staticmethod
    def train_model(model: nn.Module,
                    train_loader: DataLoader[Dict[str, torch.Tensor]],
                    val_loader: DataLoader[Dict[str, torch.Tensor]],
                    test_loader: DataLoader[Dict[str, torch.Tensor]],
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    targets_col: str = 'labels',
                    scheduler_name: str = '',
                    schedulers_kwargs: dict = {},
                    early_stop: bool = False,
                    early_stop_kwargs: dict = {},
                    num_epochs: int = 100,
                    batch_size: int = 128,
                    subset_ratio: int = 1,
                    xlabel: str = 'Epoch',
                    ylabel_acc: str = 'r2_value',
                    ylabel_loss: str = 'Loss_value',
                    title: str = 'Loss and Accuracy Plot',
                    results_savepath: str = '',
                    save_image_filename: str = 'Untitled',
                    loss_save_filename: str = 'loss_acc_file',
                    best_model_savename: str = 'best_model',
                    description_savename: str = 'description.txt',
                    # tqdm_write: bool = False,
                    standardscaler: Any = None,
                    save_model_wrt: str = 'loss', # 'loss' or 'accuracy'
                    more_description = '',
                    save_model_per_epoch: int = 50,
                    DATETIME: str = datetime_now(path=False)[0],
                    date: str = datetime_now(path=False)[1],
                    TIME: str = datetime_now(path=False)[2],
                    save_model: bool = True,
                    device: Any = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    collate_fn = None,
                    verbose: bool = True
                    ) -> tuple[List[float], nn.Module]:
        
        """
        Train a PyTorch neural network model over multiple epochs with optional validation, testing,
        learning rate scheduling, and early stopping. Logs metrics, saves plots, and stores the best model.

        ...

        Args:
            model (nn.Module): The neural network model to be trained.
            train_loader (DataLoader): Dataloader containing training data.
            val_loader (DataLoader): Dataloader containing validation data.
            test_loader (DataLoader): Dataloader containing test data.
            criterion (nn.Module): Loss function to optimize.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            targets_col (str, optional): Key for the target values in batch dictionary. Defaults to 'labels'.
            scheduler_name (str, optional): Name of the learning rate scheduler (should be in ['steplr', 'multisteplr', 'exponentiallr', 'reducelronplateau', 'cycliclr', 'onecyclelr']). Defaults to ''.
            schedulers_kwargs (dict, optional): Keyword arguments for the scheduler. Defaults to {}.
            early_stop (bool, optional): Whether to apply early stopping. Defaults to False.
            early_stop_kwargs (dict, optional): Arguments for early stopping strategy. Defaults to {}.
            num_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
            batch_size (int, optional): Batch size for dataloaders. Defaults to 128.
            subset_ratio (int, optional): Ratio of data subset for metrics during training. Defaults to 1.
            xlabel (str, optional): Label for x-axis in plots. Defaults to 'Epoch'.
            ylabel_acc (str, optional): Label for y-axis of accuracy plot. Defaults to 'r2_value'.
            ylabel_loss (str, optional): Label for y-axis of loss plot. Defaults to 'Loss_value'.
            title (str, optional): Title of the loss/accuracy plot. Defaults to 'Loss and Accuracy Plot'.
            results_savepath (str, optional): Path to save results, logs, and models. Defaults to '.'.
            save_image_filename (str, optional): Filename to save performance plot. Defaults to 'Untitled'.
            loss_save_filename (str, optional): CSV filename to save loss/accuracy logs. Defaults to 'loss_acc_file'.
            best_model_savename (str, optional): Filename for the best model checkpoint. Defaults to 'best_model'.
            description_savename (str, optional): Filename to save model configuration description. Defaults to 'description.txt'.
            DATETIME (str, optional): Timestamp string used for saving uniquely. Defaults to current time.
            tqdm_write (bool, optional): Whether to log epoch details via tqdm. Defaults to False.
            standardscaler (Any, optional): Scaler to be stored with model if applicable. Defaults to None.
            save_model_wrt (str, optional): Criteria to save best model: 'loss' or 'accuracy'. Defaults to 'loss'.
            more_description (str, optional): Additional model description for documentation. Defaults to ''.
            device (torch.device, optional): Device to use for computation (CPU/GPU). Defaults to CUDA if available.

        Returns:
            tuple:
                - List[float]: A list containing training loss values for each epoch.
                - nn.Module: The best-performing model according to validation metric.

        Raises:
            NameError: If `save_model_wrt` is not 'loss' or 'accuracy'.
        """
        
        if not results_savepath:
            results_savepath = f"./cache/{date}/{TIME}"
        models_savepath = create_folder(f"{results_savepath}/models")
        best_model_savepath = f"{models_savepath}/{best_model_savename}"
        
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        num_val = len(val_loader.dataset)
        
        out =  (
                    f"{'='*60}\n"
                    f"## NN_Trainer Configuration ##\n"
                    f"{'-'*60}\n"
                    f"{'    Date & Time':35}: {DATETIME}\n"
                    f"{'    torch_version':35}: {torch.__version__}\n"
                    f"{'    platform':35}: {platform.platform()}\n"
                    f"{'    OS':35}: {platform.system()} {platform.release()}\n"
                    f"{'    Python Version':35}: {platform.python_implementation()} {platform.python_version()}\n"
                    f"{'    Python Build':35}: {platform.python_build()}\n"
                    f"{'    Python Compiler':35}: {platform.python_compiler()}\n"
                    f"{'    Python Architecture':35}: {platform.architecture()}\n"
                    f"{'    Hostname':35}: {platform.node()}\n"
                    f"{'-'*60}\n"
                    f"{'    Model':35}: {model.__class__.__name__}\n"
                    f"{'    Device':35}: {device}\n"
                    f"{'    Epochs':35}: {num_epochs}\n"
                    f"{'    Batch Size':35}: {batch_size}\n"
                    f"{'    Optimizer':35}: {optimizer.__class__.__name__}\n"
                    f"{'    Loss Function':35}: {criterion.__class__.__name__}\n"
                    f"{'    Scheduler':35}: {scheduler_name if scheduler_name else 'None'}\n"
                    f"{'    Scheduler Arguments':35}: {schedulers_kwargs if scheduler_name else 'N/A'}\n"
                    f"{'    Early Stopping':35}: {'Enabled' if early_stop else 'Disabled'}\n"
                    f"{'    Early Stop Arguments':35}: {early_stop_kwargs if early_stop else 'N/A'}\n"
                    f"{'    Results Save Path':35}: {results_savepath}\n"
                    f"{'    Save Model Based On':35}: {save_model_wrt}\n"
                    f"{'    Best Model File Path':35}: {best_model_savepath}\n"
                    f"{'    Output Plot Filename':35}: {save_image_filename}.png\n"
                    f"\n"
                    f"{'-'*60}\n"
                    f"{'## Dataset Sizes ##'}\n"
                    f"{'-'*60}\n"
                    f"{'    Train Set':35}: {num_train} samples\n"
                    f"{'    Validation Set':35}: {num_val} samples\n"
                    f"{'    Test Set':35}: {num_test} samples\n\n"
                    f"{'    Train-Test Subset':35}: {int(num_train * subset_ratio)} samples\n"
                    f"{'    Val-Test Subset':35}: {int(num_val * subset_ratio)} samples\n"
                    f"{'    Test-Test Subset':35}: {int(num_test * subset_ratio)} samples\n"
                    f"{'-'*60}\n"
                    f"{'## Representation ##'}\n"
                    f"{'-'*60}\n"
                    f"{repr(model)}\n"
                    f"{'-'*60}\n"

                )
        out = out + more_description +'\n' + '='*80 
        description_savepath = f'{results_savepath}/{description_savename}'
        with open(description_savepath, 'w') as fp:
            fp.write(out)
        print(f"[Info] Model configuration saved to: {description_savepath}")
        
        model.to(device)
        epoch_train_losses = [None]*num_epochs
        epoch_val_losses = [None]*num_epochs
        epoch_test_losses = [None]*num_epochs
        
        train_acc_list = [None]*num_epochs
        test_acc_list = [None]*num_epochs
        val_acc_list = [None]*num_epochs
        
        lr_list = [None]*num_epochs
        time_list = [None]*num_epochs
        
        model_acc = -float('inf')
        model_loss = float('inf')
        
        train_loader_subset, _ = subset_loader(train_loader, batch_size=batch_size, subset_ratio=subset_ratio, collate_fn=collate_fn)
        test_loader_subset, _ = subset_loader(test_loader, batch_size=batch_size, subset_ratio=subset_ratio, collate_fn=collate_fn)
        val_loader_subset, _ = subset_loader(val_loader, batch_size=batch_size, subset_ratio=subset_ratio, collate_fn=collate_fn)

        print('+'*70)
        print(f'Number of Test-Train Samples: {len(train_loader_subset.dataset)}')
        print(f'Number of Test-Val Samples: {len(val_loader_subset.dataset)}')
        print(f'Number of Test-Test Samples: {len(test_loader_subset.dataset)}')
        print('+'*70)
        
        # LR Schduler
        scheduler = Trainer_functions.get_scheduler(scheduler_name=scheduler_name,
                                optimizer=optimizer,
                                **schedulers_kwargs
                                ) if scheduler_name and schedulers_kwargs else None
        
        # Initializing Early Stopping
        early_stop_kwargs={**early_stop_kwargs, 'type_':save_model_wrt}
        early_stopping = EarlyStopping(**early_stop_kwargs) if early_stop and early_stop_kwargs else None
        progress_bar = tqdm(range(num_epochs), desc="Training", unit="Eps")

        for epoch in progress_bar:
            if verbose:
                tqdm.write(f"-"*80)
                tqdm.write(f"Epoch - {epoch + 1}/{num_epochs}:")
            model.train()
            # init = time.time()
            epoch_train_loss = Trainer_functions.train_1epoch(model,
                                            train_loader,
                                            criterion,
                                            optimizer,
                                            device=device,
                                            targets_col=targets_col,
                                            description = 'Training',
                                            verbose=verbose
                                            )
            train_acc, epoch_train_loss = Trainer_functions.train_test_model(model=model, test_loader=train_loader_subset,
                                                        criterion=criterion, device=device, targets_col=targets_col,
                                                        description='Test-on-Train', verbose=verbose)
            test_acc, epoch_test_loss = Trainer_functions.train_test_model(model=model, test_loader=test_loader_subset,
                                                        criterion=criterion, device=device, targets_col=targets_col,
                                                        description='Test-on-Test', verbose=verbose)
            val_acc, epoch_val_loss = Trainer_functions.train_test_model(model=model, test_loader=val_loader_subset,
                                                        criterion=criterion, device=device, targets_col=targets_col,
                                                        description='Test-on-Val', verbose=verbose)

            train_acc_list[epoch] = round(train_acc, 4)
            test_acc_list[epoch] = round(test_acc, 4)
            val_acc_list[epoch] = round(val_acc, 4)
            lr_list[epoch] = optimizer.param_groups[0]['lr']
            time_list[epoch] = datetime_now(path=False)[0]
            
            epoch_train_losses[epoch] = round(epoch_train_loss, 6)
            epoch_test_losses[epoch] = round(epoch_test_loss, 6)
            epoch_val_losses[epoch] = round(epoch_val_loss, 6)
            
            if save_model_wrt == 'loss':
                condition_flag = epoch_val_loss < model_loss
            elif save_model_wrt == 'accuracy':
                condition_flag = val_acc > model_acc
            else:
                raise NameError('Condition name not known')

            if condition_flag:
                if verbose:
                        print(f"\033[31m### Best model -> | Epoch: {epoch + 1} | Val loss: {epoch_val_loss:.4f} | Val accuracy: {val_acc:.4f} | \033[0m")
                if save_model:
                    # Save the best model
                    Trainer_functions.save_network(model=model, 
                                optimizer=optimizer,
                                epoch=epoch,
                                device = device,
                                standardscaler=standardscaler,
                                
                                train_acc=train_acc_list[:epoch+1],
                                test_acc=test_acc_list[:epoch+1],
                                val_acc=val_acc_list[:epoch+1],
                                
                                train_loss=epoch_train_losses[:epoch+1],
                                test_loss=epoch_test_losses[:epoch+1],
                                val_loss=epoch_val_losses[:epoch+1],
                                
                                network_save_filename=best_model_savepath
                                )
                best_train_loss, best_test_loss, best_val_loss = epoch_train_loss, epoch_test_loss, epoch_val_loss
                best_train_acc, best_test_acc, best_val_acc = train_acc, test_acc, val_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model)
                model_loss = epoch_val_loss
                model_acc = val_acc

            loss_dict = OrderedDict({'epoch': list(range(1, epoch+2)), 
                                    'Time': time_list[:epoch+1],
                                    'lr': lr_list[:epoch+1],
                                    'train_loss':epoch_train_losses[:epoch+1],
                                    'test_loss':epoch_test_losses[:epoch+1],
                                    'val_loss':epoch_val_losses[:epoch+1],
                                    
                                    'train_acc': train_acc_list[:epoch+1],
                                    'test_acc': test_acc_list[:epoch+1],
                                    'val_acc': val_acc_list[:epoch+1],
                                    })
            Trainer_functions.save_metrics_csv(data_dict=loss_dict, save_csv_filename=f"{results_savepath}/{loss_save_filename}.csv")
            
            if (epoch % save_model_per_epoch == 0):
                
                model_save_name = f"{models_savepath}/model_at_epoch_{epoch}"
                if save_model:
                    if verbose:
                        print(f"Saving model at epoch {epoch + 1} to: {model_save_name}")
                    Trainer_functions.save_network(model=model, 
                                                optimizer=optimizer,
                                                epoch=epoch,
                                                device = device,
                                                standardscaler=standardscaler,
                                                
                                                train_acc=train_acc_list[:epoch+1],
                                                test_acc=test_acc_list[:epoch+1],
                                                val_acc=val_acc_list[:epoch+1],
                                                
                                                train_loss=epoch_train_losses[:epoch+1],
                                                test_loss=epoch_test_losses[:epoch+1],
                                                val_loss=epoch_val_losses[:epoch+1],
                                                
                                                network_save_filename=model_save_name)
            
            # Update tqdm progress bar with accs
            if verbose:
                tqdm.write(f"At {time_list[epoch]} |Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | "
                        f"Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Test Acc: {test_acc:.2f}")
            
            # # Set postfix in tqdm bar for live update
            if not verbose:
                progress_bar.set_postfix({'Epoch':f"{epoch+1}/{num_epochs}", "Train Loss": f"{epoch_train_loss:.4f}", "Val Loss": f"{epoch_val_loss:.4f}",
                                          "Test Loss": f"{epoch_test_loss:.4f}", "Train Acc": f"{train_acc:.4f}", "Val Acc": f"{val_acc:.4f}", "Test Acc": f"{test_acc:.4f}"})
                
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_val_loss)
                else:
                    scheduler.step()
            
            if early_stop:  
                early_stop_val = val_acc if save_model_wrt=='accuracy' else  epoch_val_loss
                early_stopping(early_stop_val)
                if early_stopping.early_stop:
                    print(f"Early stop triggered at epoch: {epoch + 1}")
                    print('='*80)
                    break
        print(f'Best Model Details:\n  Epoch: {best_epoch+1}\n    Train Loss: {best_train_loss}, Test Loss: {best_test_loss}, Val Loss: {best_val_loss}\n   \
            Train Acc: {best_train_acc}, Test Acc: {best_test_acc}, Val Acc: {best_val_acc}')
                
        Trainer_functions.plot_loss_save_images(train_acc=train_acc_list, 
                                                test_acc=test_acc_list,
                                                val_acc=val_acc_list,
                                                train_loss=epoch_train_losses, 
                                                test_loss=epoch_test_losses,
                                                val_loss=epoch_val_losses,
                                                xlabel = xlabel,
                                                ylabel_acc = ylabel_acc,
                                                ylabel_loss = ylabel_loss,
                                                title = title,
                                                save_image_filename = f"{results_savepath}/{save_image_filename}"
                                               )

        print('Training completed.')
        
        return epoch_train_losses, best_model



    @staticmethod
    def plot_loss_save_images(train_acc: List[float],
                            test_acc: List[float],
                            val_acc: List[float],
                            train_loss: List[float],
                            test_loss: List[float],
                            val_loss: List[float],
                            xlabel: str = 'Epoch',
                            ylabel_loss: str = 'Loss Value',
                            ylabel_acc: str = 'r2_value',
                            title: str = 'Loss and Accuracy Plot',
                            save_image_filename: str = '',
                            marker: str = '', 
                            markersize: int = 5, 
                            linewidth: int = 2, 
                            linestyle: str = '-', 
                            title_fontsize: int = 25, 
                            xyticks_fontsize: int = 10 
                            ) -> None:
        """
        Plots training, validation, and test accuracy and loss curves over epochs.

        Args:
            train_acc (List[float]): Accuracy values for training set per epoch.
            test_acc (List[float]): Accuracy values for test set per epoch.
            val_acc (List[float]): Accuracy values for validation set per epoch.
            train_loss (List[float]): Loss values for training set per epoch.
            test_loss (List[float]): Loss values for test set per epoch.
            val_loss (List[float]): Loss values for validation set per epoch.
            xlabel (str): Label for x-axis. Defaults to 'Epoch'.
            ylabel_loss (str): Label for y-axis on loss plot. Defaults to 'Loss Value'.
            ylabel_acc (str): Label for y-axis on accuracy plot. Defaults to 'r2_value'.
            title (str): Title for the entire figure.
            save_image_filename (str): Path to save plot (without extension).
            marker (str): Marker symbol for line plot.
            markersize (int): Size of the marker.
            linewidth (int): Width of the lines in the plot.
            linestyle (str): Style of the lines.
            title_fontsize (int): Font size for titles.
            xyticks_fontsize (int): Font size for tick labels.

        Returns:
            None. Saves plot as PNG file and closes the figure.
        """
        
        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8)) 
        fig.suptitle(title, fontsize=title_fontsize)  
        
        # Plot for accuracy
        axs[0].plot(
            range(len(train_acc)), train_acc, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='train',
            color='green'  
        )
        axs[0].plot(
            range(len(test_acc)), test_acc, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='test',
            color='red'  
        )
        axs[0].plot(
            range(len(val_acc)), val_acc, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='val',
            color='blue'  
        )
        axs[0].set_xlabel(xlabel, fontsize=title_fontsize-5)
        axs[0].set_ylabel(ylabel_acc, fontsize=title_fontsize-5)
        axs[0].grid(linestyle='--')
        axs[0].legend(loc='upper right', fontsize=xyticks_fontsize)
        axs[0].set_title('Epoch vs Accuracy', fontsize=title_fontsize-5)
        axs[0].tick_params(axis='both', labelsize=xyticks_fontsize) 
        
        # Plot for loss
        axs[1].plot(
            range(len(train_loss)), train_loss, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='train',
            color='green' 
        )
        axs[1].plot(
            range(len(test_loss)), test_loss, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='test',
            color = 'red' 
        )
        axs[1].plot(
            range(len(val_loss)), val_loss, 
            marker=marker, 
            markersize=markersize, 
            linewidth=linewidth, 
            linestyle=linestyle, 
            label='val',
            color = 'blue' 
        )
        axs[1].set_xlabel(xlabel, fontsize=title_fontsize-5)
        axs[1].set_ylabel(ylabel_loss, fontsize=title_fontsize-5)
        axs[1].grid(linestyle='--')
        axs[1].legend(loc='upper right', fontsize=xyticks_fontsize)
        axs[1].set_title('Epoch vs Loss', fontsize=title_fontsize-5)
        axs[1].tick_params(axis='both', labelsize=xyticks_fontsize) 
        
        plt.tight_layout() 
        if save_image_filename:
            plt.savefig(f"{save_image_filename}.png", bbox_inches='tight')
            print(f'\033[1;35mSaved loss accuracy plot for Train-Test-Val data: "{f"{save_image_filename}.png"}"\033[0m')
        
        plt.close() 
        
        
        
    @staticmethod
    def save_network(model: nn.Module = None,
                    optimizer: torch.optim.Optimizer = None,
                    epoch : int = None,
                    learning_rate : float = None,
                    train_acc: List[float] = [],
                    test_acc: List[float] = [],
                    val_acc: List[float] = [],
                    train_loss: List[float] = [],
                    test_loss: List[float] = [],
                    val_loss: List[float] = [],
                    standardscaler: Any = None,
                    device: Any = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    network_save_filename: str = 'network_best'
                    ) -> None:

        """
        Saves the model checkpoint, optimizer state, training metrics, and source file content.

        Args:
            model (nn.Module): Trained model to be saved.
            optimizer (torch.optim.Optimizer): Optimizer used during training.
            epoch (int): Epoch at which the model is saved.
            learning_rate (float): Learning rate used for training.
            train_acc (List[float]): Training accuracy values.
            test_acc (List[float]): Test accuracy values.
            val_acc (List[float]): Validation accuracy values.
            train_loss (List[float]): Training loss values.
            test_loss (List[float]): Test loss values.
            val_loss (List[float]): Validation loss values.
            standardscaler (Any): Optional scaler object used on input data.
            network_save_filename (str): Name for the saved checkpoint file (without extension).

        Returns:
            None. Saves a `.pt` checkpoint file to disk.
        """
        network_save_filename = ''.join([network_save_filename, '.pt'])
        print(f'\033[1;95mSaving network in: "{network_save_filename}"\033[0m')
        
        source_file = os.path.abspath(__file__)
        with open(source_file, 'rb') as fp:
            file_ = fp.read()
            utils_pyfile = pickle.dumps(file_)
            
        torch.save({'model_class': model.__class__.__name__,
                    'torch_version': torch.__version__,
                    'python_version': platform.python_version(),
                    'device': str(device),
                    'hostname': platform.node(),
                    'epoch': epoch+1,
                    'datetime':datetime_now(),
                    'utils_pyfile': utils_pyfile,
                    'network_pyfile': getattr(model, 'model_script', None),
                    'learning_rate' : learning_rate,
                    'standardscaler':standardscaler,
                    
                    'train_loss': train_loss,
                    'val_loss': val_loss, 
                    'test_loss': test_loss, 
                    
                    'train_acc': train_acc,
                    'val_acc': val_acc, 
                    'test_acc': test_acc, 
                    
                    'state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer': copy.deepcopy(optimizer.state_dict()),
                    'representation': str(repr(model).replace('\n', ''))
                    }, network_save_filename)
    #------------------------------------------------------

    #save (csv) different losses
    @staticmethod
    def save_metrics_csv(data_dict: dict,
                        save_csv_filename: str = 'temp_metrics.csv'):
        """
        Saves metrics from training/validation/testing to a CSV file.

        Args:
            data_dict (dict): Dictionary where keys are metric names (e.g., 'train_loss') 
                            and values are lists or tuples of metric values.
            save_csv_filename (str): Name of the CSV file to save the metrics.

        Returns:
            pd.DataFrame: DataFrame containing the saved metrics.
        """
        if not data_dict:
            raise ValueError("No data provided to save.")

        for key, value in data_dict.items():
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"Value for '{key}' must be list or tuple, not {type(value).__name__}")

        df = pd.DataFrame(data_dict)
        df.to_csv(save_csv_filename, index=False)
        
        return df

        
    # pre-trained network load function for test
    @staticmethod
    def load_test_network(network: nn.Module,
                        temp_network_path: str = '',
                        strict: bool = True,
                        device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        verbose:bool = True
                        ):
        """
        Loads a previously saved model checkpoint for evaluation or resumption.

        Args:
            network (nn.Module): Model architecture to load weights into.
            optimizer (torch.optim.Optimizer): Optimizer to load state into.
            temp_network_path (str): Path to the `.pt` checkpoint file.
            device (str): Device to map the checkpoint onto (e.g., 'cuda' or 'cpu').

        Returns:
            [model_repr, standardscaler, checkpoint, network_pyfile]
            Prints error if checkpoint is missing or corrupted.
        """
        
        if os.path.isfile(temp_network_path):
            if verbose:
                print('-'*80)
                print('Loading pre-trained network checkpoint from: "{}"'.format(temp_network_path))
            checkpoint = torch.load(temp_network_path, map_location=device, weights_only=False)
            #--------------------------------------------------
            epoch_best_network = checkpoint.get('epoch', 'UNKNOWN')
            train_loss = checkpoint.get('train_loss', 'NA')
            val_loss = checkpoint.get('val_loss', 'NA')
            test_loss = checkpoint.get('test_loss', 'NA')
            train_acc = checkpoint.get('train_acc', 'NA')
            val_acc = checkpoint.get('val_acc', 'NA')
            test_acc = checkpoint.get('test_acc', 'NA')
            network_pyfile = checkpoint.get('network_pyfile', 'NA')
            model_class = checkpoint.get('model_class', 'NA')
            standardscaler = checkpoint.get('standardscaler', False)
            model_repr = checkpoint.get('representation', False)
            #--------------------------------------------------
            #+++++++++++++++++++++++++++++++++++++++++++++++++++
                
            network.load_state_dict(checkpoint['state_dict'], strict=strict)
            if verbose:
                print('Loaded pre-trained network checkpoint from "{}"\nepoch: {} train loss: {} val loss: {} test loss: {} train acc: {} val acc: {} test acc: {} ' \
                    .format(temp_network_path, epoch_best_network, train_loss[-1], val_loss[-1], test_loss[-1], train_acc[-1], val_acc[-1], test_acc[-1])
                        )
                print('='*80)
            
            return model_repr, standardscaler, checkpoint, network_pyfile
        else:
            print('-'*80)
            print(f'No pre-trained network checkpoint found at "{temp_network_path}"')
            print('='*80)
        