# Predicting Compound-Protein Interaction using Hierarchical Graph Convolutional Networks
This repository contains source code and data for the article "Predicting Compound-Protein Interaction using Hierarchical Graph Convolutional Networks".

### Requirements
The source code is written in Python and requires some packages:
* Python 3 
* PyTorch 
* Torch Geometric 
* RDKit 
* h5py
* NetworkX

### Data 
Input data need to be pre-processed before using our program. The pre-processing step prepares hierarchical representation for small molecules and saves all neccessary information into a h5py file and a meta file (if it is training data). The h5py file allows the program to work with large data sets. The input data files which contain lines of tuples (smiles, protein sequence, label) can be pre-processed using provided scripts, in which: 
- ```convert_data.py```: for training data or splitting training, testing data 
- ```convert_testing_data.py```: only for testing data

The examples of input data files and the ouput after pre-processing can be found in folder ```data/bindingdb``` 

Note: As training dataset of ```chembl27``` is large, we split them into 4 separate files. You should merge them into one file before running ```convert_data.py``` script. 

### Source code
Running the script ```train_gnncnn_cpi_model.py``` to train (and to test after training) a new model. For example:
<pre><code class="language-python"> python train_gnncnn_cpi_model.py --train data/bindingdb/bindingdb.gnn.train --test data/bindingdb/bindingdb.gnn.test --val data/bindingdb/bindingdb.gnn.val --root data/bindingdb/ --lr 0.0001 --nloop 50 --output data/bindingdb/bindingdb.gnn.model --meta data/bindingdb/bindingdb.gnn.meta --nkernel 6</code></pre>

Running the script ```test_gnncnn_cpi_model.py``` to test a trained model. For example: 
<pre><code class="language-python"> python test_gnncnn_cpi_model.py --test data/bindingdb/bindingdb.gnn.test --root data/chembl27/ --model data/bindingdb/bindingdb.gnn.model</code></pre>

Information of required arguments can be found in corresponding scripts.

If you have any further question, please contact us via email: danh.bui-thi@uantwerpen.be or kris.laukens@uantwerpen.be
