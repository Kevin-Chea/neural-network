# Neural Networks

This repository contains a simple implementation of a reusable neural network in Python.

**The work is still in progress**, but some features already work.

## XOR example
An XOR example is given. Run the following command to execute it :
```shell
python3 xor.py
```
You can go into the file and modify the number of layer and / or the number of neuron to experiment.

It is possible to save the model if you are satisfied of the result. In xor.py, uncomment the last line : 
```python
# nn.save_model("xor.pkl")
```
You can change the file name if you want.

You can load the model by calling the *load_model* method. In saved_xor.py, change the name of the file if necessary and then run the file :
```shell
python3 saved_xor.py
```
## Testing
Tests are located in the **/test** folder.
It uses unittest to run the test. If you want to run a specific file, for instance test/test_neuron.py, run the following command :
```shell
python3 -m unittest test.test_neuron
```
