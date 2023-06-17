import SimpleNN
import copy

model = SimpleNN.Network([1,2,5,2,1])

in_data = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
out_data = [[1], [4], [9], [16], [25], [36], [49], [64], [81], [100]]

print(model.predict([10]))

model.train(in_data, out_data, epochs=10000, error_samples=5)

print(model.predict([10]))
    
while True:
    print(model.predict([float(input())]))
