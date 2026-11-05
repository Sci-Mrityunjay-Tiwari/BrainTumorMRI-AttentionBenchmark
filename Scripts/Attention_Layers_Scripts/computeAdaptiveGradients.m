function [loss, gradients] = computeAdaptiveGradients(net, X, T)

    predictions = forward(net, X);

    probabilities = softmax(predictions);

    loss = crossentropy(probabilities, T);

    gradients = dlgradient(loss, net.Learnables);

end