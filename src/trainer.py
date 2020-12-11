#
# Trainer
#
import torch

from typing import List, Dict
from tqdm import tqdm
import time
import os
import datetime
from src.dag import DAG


class Trainer():
    def __init__(self,
                 steps: int,
                 network: DAG,
                 loss_fn,
                 optimizer,
                 steps_per_checkpoint: int,
                 save_path: str,
                 save_name: str,
                 cuda: bool = True):

        # Training parameters
        self.steps = steps
        self.net = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Check accelerator compatability and send the network to the compatible device
        self.device = _check_cuda(cuda)
        self.net.to(self.device)

        # Model check point parameters
        self.spc = steps_per_checkpoint
        self.save_name = save_name
        self.save_path = save_path

    def _check_cuda(self, cuda: bool) -> str:
        '''
          Checks the availablity of a CUDA enabled accelerator. If one isn't available, raise an error. We dont currently allow for distributed training. This is coming in the future

          Args:
            cuda: boolean describing whether the user has an available accelerator

          Returns:
            torch_device: the string representing the device to be used by torch
        '''
           if not cuda:
                return 'cpu'
            elif not torch.cuda.is_availble():
                raise Exception('You claim to have a CUDA enabled device, but pytorch cannot find it')
            else:
              return 'cuda:0'

    def _train_step(self, inputs: Dict[str, torch.tensor]) -> torch.tensor:
        '''
          Runs a single training step a single batch of inputs.

          Args:
            inputs: This is a dictionary of the inputs of the given DAG. This should match the naming scheme of the defined DAG

          Returns:
            loss: returns outputs of the network run
        '''
        pass

    def train(self, dataloader):
        '''
        This function handles all of training logic

        Args: 
          dataloader This should a preloaded pytorch dataloader

        '''
        start_time = time.time()        
        epoch = 0
        step = 0

        while step < self.steps:
          epoch += 1
          print(f'Starting Epoch {epoch}')

          with tqdm(enumerate(dataloader)) as t
            for batch_num, batch_tuple in t:

              batch, labels = batch_tuple

              self.optimizer.zero_grad()

              # Cuda enable any input tensors if using a cuda device and then send all the data to the proper device
              if self.device != 'cpu':
                [batch[k].cuda() for k in batch]
              [batch[k].to(self.device) for k in batch]

              loss = self.creiterion(_train_step(batch), labels)
              loss.backward()
              self.optimizer.step()

              t.set_description('Step: %6d Epoch: %4d Batch: %4d Loss: %.3f' %(step + 1, epoch, batch_num, loss.item()))

          step +=1

          # Save the model dictionary in the path {save_path}/save_name_{step number}
          if step % self.spc == 0:
            torch.save(self.net, os.path.join(self.save_path, self.save_name + str(step)))
          
        print(f'Training has finished. Completed in {datetime.timedelta(seconds=(time.time() - start_time))}.')
