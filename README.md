# SimpleNN

### Oversimplified Neural Network for educational purposes 

Educational oversimplified implementation of a neural network for learning the basics of how AI can predict and be trained. 

## Usage

#### Prerequisites: Python 3 (3.5 and higher recommended)

A neural network can be created by initializing a <code>SimpleNN.Network</code> object with a the network structure as a list as an argument.
Each list item represents one layer, and the value of the list item represents the number of neurons in that layer.

#### Example:

    import SimpleNN

    model = SimpleNN.Network([1,2,1])

This code initializes a Network with 3 layers. The input layer contains one neuron, so only one input value is accepted. 
The second layer adds two hidden neurons to the network. These do neither receive inputs directly nor output values directly, as they are only used to process data from the previous layer.
Lastly, the Network contains one output neuron, so exactly one value is outputted.

### Getting model output

    model.predict([inputs])

Network.predict() will output the predicted output for the given inputs. The length of the input list always has to be equal to the number of input neurons (first layer).

### Model training

    model.train(inputs, outputs, epochs=10, step_size_threshold=5, step_size_override=0, verbose=True, error_samples=0)

Network.train() will train the model using the specified input and desired output data for the specified number of epochs.

#### Arguments:

<code>inputs</code>
List of input data (list of lists of integers or floats)

<code>outputs</code>
List of desired output data (list of lists of integers or floats)
The index in the inputs list corresponds to the index in the ouputs list

<code>epochs</code>
Number of training cycles of the given data

<code>step_size_threshold</code>
Number of unsuccessful epochs before the step size is decreased by the factor of 10

<code>step_size_override</code>
Force a specific step size while training (0 = automatic)

<code>verbose</code>
Setting <code>verbose=True</code> outputs training details to the console. If <code>verbose</code> is set to <code>False</code>, this information is not displayed.

<code>error_samples</code>
Number of additional training samples being used for accuracy evaluation 