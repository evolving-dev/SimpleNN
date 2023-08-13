import SimpleNN

#Create simplified neural network with one input node and one output node (no hidden nodes)
model = SimpleNN.Network([1,1])

#Dataset that takes in a number and multiplies it by 2
in_data = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
out_data = [[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]]

test_value = 20


print("Created a new untrained neural network. Its predictions are random right now, because it has not seen any training data yet. ")

prediction = model.predict([test_value])[0]

print("The untrained model predicts that " + str(test_value) + " * 2 = " + str(prediction) + " (difference to truth: " + str(prediction - test_value * 2) + ")")

for i in range(10):
    print("\nTraining for 10 epochs...")
    model.train(in_data, out_data, epochs=10, verbose=False)
    prediction = model.predict([test_value])[0]
    print("The model now predicts that " + str(test_value) + " * 2 = " + str(prediction) + " (difference: " + str(prediction - test_value * 2) + ")")

print("\nSaving model...")
model.save_to("example_model.json")

while True:
    print("\n")
    print(model.predict([float(input("Enter a number for the model to predict: "))])[0])
