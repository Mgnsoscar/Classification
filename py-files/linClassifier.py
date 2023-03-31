import numpy as np
from Dataset import *
import matplotlib.pyplot as plt

def sigmoidFunction(x):     # Applies the sigmoid function to all values of input data x
    return 1 / (1 + np.exp(-x))

def getPredictionVectors(sampleVectors, weightingMatrix):

    # This function will multiply our sample vector with our weighting matrix.
    # Since we need our resulting vector values to be between 0 and 1, we apply the sigmoid function to
    # the results of each matrix multiplication.
    # See equation (3.20) from the compendium on classification

    # The function will return an array of prediction vectors, all corresponding to their input sample vector
    # In mathematical terms it returns an array of
    # the calculation below for each sample vector:

    #                               1
    #           -----------------------------------------
    #            -(sample vector * weighting matrix)
    #           e                                    + 1

    # The resulting vectors should be in the form of [0<x<1, 0<y<1, 0<z<1]^T, where x, y, and z corresponds to the
    # different classes/species of the Iris flower. The greatest value decides what Iris flower the sample will be
    # classified as.
    string = """
    matrixMultiplicationResults =   []  # Creates list where the results from the matrix multiplication will go

    for vector in sampleVectors:  # Iterate through every sample vector

        result  =   np.matmul( weightingMatrix, vector )  # Perform the matrix multiplication
        matrixMultiplicationResults.append( result )      # Append the result of the multiplication to the results list

    matrixMultiplicationResults     =   np.array( matrixMultiplicationResults )  # Convert into numpy array

    predictionVectors = sigmoidFunction( matrixMultiplicationResults )  # Apply the sigmoid function to the results
    print(predictionVectors)
    return predictionVectors  # Return the prediction vectors
    """
    exponentials = np.array([np.exp(-(np.matmul(weightingMatrix, sample))) for sample in sampleVectors])
    denominators = exponentials + 1
    predictions = 1 / denominators
    print(predictions)
    return predictions

def getPredictions(predictionVectors):

    # This function takes in an array with prediction vectors and rounds them up to make a prediction.
    # For example if the input prediction vector is [0.45, 0.76, 0.92]^T, the output of this function will
    # be [0, 0, 1]^T, meaning the sample has been classified as the Iris flower corresponding to the last index.

    predictionIndexes       =   np.argmax(predictionVectors, axis = 1)  # Makes list with indexes of the greatest value in each vector
    predictions             =   []  # List with rounded prediction vectors

    for index in predictionIndexes:      # Iterate through every index of greatest value in each prediction vector

        # Makes a vector where the value corresponding to the index in predictionIndexes is 1 and the others 0
        roundedVector = np.array([ i == index for i in range(3) ], dtype=np.uint8)

        predictions.append(roundedVector)   # Adds the rounded vector to the list of predictions

    predictions     =   np.array( predictions )     # Converts to numpy array

    return predictions  # Returns the list of all processed prediction vectors

def updateWeigthingMatrix(predictionVectors, vectorPairs, currentMatrix, stepFactor=0.01):

#                                    Implements eq. (3.22) from the compendium on classification
#
#             len(samples)
#                 ___
#     ∇_W MSE =   \                                                                                                    T
#                 /__  [ (predictions[k] - rightanswers[k]) ⚬ predictions[k] ⚬ (1 - predictions[k]) ] * sampleVectors[k]
#                 k=1
#
# This calculates the gradient of the Mean Square Error with respect to W. Result will be a vector pointing in the
# direction of greatest ascent. Here x[]] is sample vector[k]

#                                    Implements eq. (3.23) from the compendium on classification
#
#           updated Weigthing matrix    =   current weigthing matrix  - ( alpha * delta_W MSE )
#
# This takes the current weigthing matrix and nudges it in the oposite direction of the MSE gradient. How much it
# is shifted is decided by the step factor alpha.

    actualClass     =   vectorPairs[1]
    sampleVectors   =   vectorPairs[0]
    string = """
    num_features = len(sampleVectors[0]) - 1  # Subtract 1 because of the 1-fill
    num_classes = 3
    grad_g_MSE = predictionVectors - actualClass  # dim (30,3)
    grad_z_g = predictionVectors * (1 - predictionVectors)  # dim (30,3)

    grad_W_z = np.array([np.reshape(sample, (1, num_features + 1)) for sample in sampleVectors])

    sums = []
    for k in range(len(grad_g_MSE)):
        result = np.matmul( np.reshape( grad_g_MSE[k] * grad_z_g[k], (num_classes, 1) ), grad_W_z[k])
        sums.append(result)
    gradient = np.sum(sums)

    next_W = currentMatrix - stepFactor * gradient

    return next_W
    """
    num_features = len(sampleVectors[0]) - 1 # Subtract 1 because of the 1-fill
    num_classes = 3
    grad_g_MSE = predictionVectors - actualClass # dim (30,3)
    grad_z_g = predictionVectors * (1 - predictionVectors) # dim (30,3)

    grad_W_z = np.array([ np.reshape(sample, (1, num_features+1)) for sample in sampleVectors ])

    grad_W_MSE = np.sum( np.matmul(np.reshape(grad_g_MSE[k] * grad_z_g[k], (num_classes, 1)), grad_W_z[k]) for k in range(len(grad_g_MSE)) )

    next_W = currentMatrix - stepFactor * grad_W_MSE

    return next_W

def getMSE(predictions, rightAnswers):

    # Implements eq. (3.19) using
    # predicted_label_vectors[k] = g_k
    # true_label_vectors[k] = t_k

    error   =   predictions - rightAnswers
    MSE     =   np.sum( np.matmul( np.transpose(error) , error ) ) / 2

    return MSE

def getErrorRate(predictions, rightAnswers):

    samples     =   len(rightAnswers)
    errors      =   0

    for i in range(len(rightAnswers)):

        if not np.array_equal(  rightAnswers[i], predictions[i]):
            errors += 1

    errorRate   =   errors/samples

    return errorRate

def train_linear_classifier(
    trainingVectorPairs     ,   testingVectorPairs       ,
    iterations = 1000       ,   stepFactor = 0.01       ):
    ######################################################

    classes     =   3
    features    =   len(trainingVectorPairs[0][0]) - 1      # How many features in a vector minus 1 for the added 1

    MSEPrIteration          =   []
    errorRatePrIteration    =   []

    weightingMatrix = np.zeros((classes, features+1))    # Initialize classesxfeatures+1 matrix with random values

    for iteration in range(iterations):

        # Training
        trainingPredictionVectors = getPredictionVectors(trainingVectorPairs[0], weightingMatrix)

        weightingMatrix = updateWeigthingMatrix(trainingPredictionVectors,  trainingVectorPairs,
                                                weightingMatrix          ,
                                                stepFactor)
        # Testing
        testingPredictionVectors    =   getPredictionVectors(testingVectorPairs[0], weightingMatrix)
        testingPredictions          =   getPredictions(testingPredictionVectors)

        currentMSE      =   getMSE(testingPredictions, testingVectorPairs[1])
        MSEPrIteration.append(currentMSE)

        currentErrorRate    =   getErrorRate(testingPredictions, testingVectorPairs[1])
        errorRatePrIteration.append(currentErrorRate)

    return weightingMatrix, np.array(MSEPrIteration), np.array(errorRatePrIteration)

def plotMSE(trainingVectorPairs, testingVectorPairs, alphas, iterations=10):

    MSEprAlpha = []

    for alpha in alphas:
        _,  MSEprIteration,  _      =   train_linear_classifier(trainingVectorPairs,    testingVectorPairs  ,
                                                                iterations         ,    alpha               )
        MSEprAlpha.append(MSEprIteration)

    for i in range(len(alphas)):
        MSE = MSEprAlpha[i]
        alpha = alphas[i]

        iteration_numbers = range(len(MSE))
        plt.plot(iteration_numbers, MSE, label='$\\alpha={' + str(alpha) + '}$')

    plt.xlabel("Iteration number")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    return


training, testing = makeVectorPairs(30)
plotMSE(training, testing, [0.0025, 0.005, 0.0075, 0.01])