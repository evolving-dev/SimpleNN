import copy
import random
import time

class Neuron:
    def __init__(self, type="neuron", prev_layer=1):
        self.type = str(type)

        self.input_weights = [random.random() for i in range(prev_layer)]

    def __repr__(self):
        if self.type == "neuron":
            return "<Neuron with input weights of " + str(self.input_weights) + ">"
        else:
            return "< " + str(self.type) + " Neuron with input weights of " + str(self.input_weights) + ">"

class Network:
    def __init__(self, layers=[], neurons=[]):
        self.layers = layers
        self.layer_count = len(layers)
        self.neurons = []

        self.step_size = 10

        for layer_no, layer in enumerate(layers):
            self.neurons += [[]]
            for neuron in range(layer):
                if layer_no == 0:
                    self.neurons[-1] += [Neuron(type="input")]
                else:
                    self.neurons[-1] += [Neuron(prev_layer=layers[layer_no - 1])]

    def get_architecture(self):
        return str(self.neurons)

    def get_layer_output(self, data, layer_no):
        if (len(data) != self.layers[layer_no]) and (len(data) != len(self.neurons[layer_no][0].input_weights)):
            raise IndexError("Data length of " + str(len(data)) + " does not match layer length of " + str(self.layers[layer_no]) + ".")

        if self.layer_count - 1 < layer_no:
            raise IndexError("Layer " + str(layer_no) + " does not exist. Maximum value is " + str(self.layer_count - 1))

        neuron_outputs = []

        for neuron in self.neurons[layer_no]:
            neuron_inputs = data.copy()

            for element, value in enumerate(neuron_inputs):
                neuron_inputs[element] = value * neuron.input_weights[element] #Apply input weights

            neuron_output = sum(neuron_inputs)

            neuron_outputs += [neuron_output]

        return neuron_outputs

    def predict(self, data):

        last_output = data.copy()

        if len(data) != self.layers[0]:
            raise IndexError("Input length of " + str(len(data)) + " does not match input layer length of " + str(self.layers[0]) + ".")

        for layer_no in range(self.layer_count):
            last_output = self.get_layer_output(last_output, layer_no)

        return last_output

    def run_training_sample(self, data, train_output):
        actual_output = self.predict(data)
        differences = [abs(trained - actual) for trained, actual in zip(train_output, actual_output)]
        total_difference = sum(differences)

        derived_network = copy.deepcopy(self)

        random_layer = random.randint(0, self.layer_count - 1)
        random_neuron = random.randint(0, len(self.neurons[random_layer]) - 1)

        random_position = random.randint(0, len(derived_network.neurons[random_layer][random_neuron].input_weights) - 1)

        derived_network.neurons[random_layer][random_neuron].input_weights[random_position] += random.choice([-1, 1]) * self.step_size

        derived_output = derived_network.predict(data)
        derived_differences = [abs(trained - actual) for trained, actual in zip(train_output, derived_output)]
        derived_total_difference = sum(derived_differences)

        if derived_total_difference <= total_difference:
            if derived_total_difference == total_difference and random.random() < 0.95: #If change didn't make a difference, keep it with a 5% chance
                return False
            self.neurons[random_layer][random_neuron] = copy.deepcopy(derived_network.neurons[random_layer][random_neuron])
            return True

        return False

    def run_training_epoch(self, inputs, outputs):
        if len(inputs) != len(outputs):
            raise IndexError("Input data count does not match output data count.")

        changes = False

        for batch_no in range(len(inputs)):
            trained = self.run_training_sample(inputs[batch_no], outputs[batch_no])

            if trained:
                changes = True

        return changes

    def train(self, inputs, outputs, epochs=10, step_size_threshold=5, step_size_override=0, verbose=True, error_samples=0):
        changes = False
        epochs_without_progress = 0

        self.step_size = self.step_size
        if step_size_override != 0:
            self.step_size = step_size_override

        for epoch_no in range(epochs):
            t = time.time()
            trained = self.run_training_epoch(inputs, outputs)

            if trained:
                changes = True
                epochs_without_progress = 0
            else:
                epochs_without_progress += 1

            if epochs_without_progress > step_size_threshold and step_size_override == 0:
                self.step_size /= 10
                epochs_without_progress = 0
                if verbose:
                    print("No progress for " + str(step_size_threshold) + " epochs. Reducing step size to " + str(self.step_size) + ".")

            if error_samples > 0:
                error_rate = 0
                for i in range(error_samples):
                    random_test_item = random.randint(0, len(inputs) - 1)
                    model_output = self.predict(inputs[random_test_item])
                    difference = sum([abs(trained - actual) for trained, actual in zip(model_output, outputs[random_test_item])])
                    error_rate += difference

                error_rate /= error_samples



            if verbose:
                print("Training epoch " + str(epoch_no + 1) + " of " + str(epochs) + " (" + str(round((time.time() - t) * 1000, 1)) + "ms)")

            if error_samples > 0:
                print("(" + str(epoch_no + 1) + "/" + str(epochs) + ") Average total training error: " + (str(round(error_rate, 4 if error_rate > 0.1001 else 10))))

        return changes
