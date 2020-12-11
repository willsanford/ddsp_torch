#
# DAG
#

from typing import List, Dict
import torch
from torch.nn import Module


class DAGNode(Module):
    def __init__(self,
                 input_names: str,
                 output_names: str,
                 name: str):

        super(DAGNode, self).__init__():
        self.input_names = input_names
        self.output_names = output_names
        self.name = name

    def get_inputs(self) -> List[str]:
        return self.input_names

    def get_outputs(self) -> List[str]:
        return self.output_names

    def get_name(self) -> str:
        return self.name

    def forward(self, x: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        raise NotImplementedError


class DAG(Module):
    '''
      This is our wrapper of a dag
    '''

    def __init__(self,
                 dag_list: List[Module],
                 inputs: List[str],
                 outputs: List[str]):
        super(Module, self).__init__()

        self.node_list = node_list
        self.inputs = inputs
        self.outputs = outputs

        # Run a check of the DAG list
        _check_dag_list()

    def forward(x: Dict) -> Dict:
        '''
          This will run a forward pass of the dag. This input is given as dictionary of inputs. The dag will 
          sucsessively replace entries in this dictionary with dag node outputs until the dictionary only 
          includes outputs.  
        '''

        # Raise an error if the incorrect keys are given.
        if set(x.keys()) != set(self.inputs):
            raise Exception(
                f'Incorrect Inputs. The input keys that were expected were \n\t\t{self.inputs} \n\t but we got \n\t\t\ {x.keys()}')

        # create the output dictionary that we will collect the signal as it processed
        outputs = x

        for node in self.node_list:
            # Generate a dictionary of the proper inputs. Run these inputs through the node and append the outputs
            # to the greater ouput dictionary
            node_inputs = {k: outputs[k] for k in node.get_inputs}
            node_outputs = node(node_inputs)
            [outputs.pop(key) for key in node_inputs.keys()]
            outputs.update(node_outputs)

        # Raise an exception if the outputs are different from the outputs expected
        if set(outputs.keys()) != set(self.outputs):
            raise Exception(
                f'Incorrect Outputs. The output keys that were expected were \n\t\t{self.outputs} \n\t but we got \n\t\t\ {outputs.keys()}')

        return outputs

    def _check_dag_list(self):
        '''
        Internal function for checking the validity of a dag. This will run through the dag checking the input
        and output pattern. An error will be raised if the dag is not truly a dag
        '''

        outputs = {key: True for key in self.inputs}

        for node in self.node_list:
            try:
                node_inputs = {k: outputs[k] for k in node.get_inputs()}
                [outputs.pop(key) for key in node_inputs.keys()]
            except Exception as e:
                raise Exception(
                    f'The inputs of this node - {node.name} -  cannot be found in the current graph') from e

            try:
                outputs.update(node.get_outputs())
            except Exception as e:
                raise Exception(
                    'This should not be able to fail. Can\'t help you much on this one') from e

        if set(outputs.keys()) != set(self.outputs):
            raise Exception(
                f'The outputs produced by this DAG are not identical to the given outputs. We got \n\t\t{outputs.keys()} \n\t when we should have gotten \n\t\t{self.outputs}')
