# training a simple neural network
# %%
from neuraltrace.nn import MultiLayer_Perceptron

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# define the neural network
net = MultiLayer_Perceptron(3, [4, 2, 4, 1])
# %%
learning_rate = 0.01

# training loop
for epoch in range(50):

    # forward pass
    # compute predictions
    y_pred = [net(x) for x in xs]
    # compute loss
    loss = sum(((y_t - y_p) ** 2) for y_t, y_p in zip(ys, y_pred))

    # backward pass
    # flush gradients so that they don't accumulate
    for param in net.parameters():
        param.grad = 0.0
    # compute gradients
    loss.backward()

    # update parameters(gradient descent)
    for param in net.parameters():
        param.value -= learning_rate * param.grad

    # print loss
    print(f"Epoch {epoch+1}: loss = {loss.value}")

print("Training complete!")
print(y_pred)