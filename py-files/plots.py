from Dataset import *
from linClassifier import *
import matplotlib.pyplot as plt

def scatterplots(xAxis, yAxis):

# This function will create scatterplots based on the sample data from the dataset.
# The function arguments decide what values to plot against each other.

# Possible arguments are type string and as follows:
    # "Sepal length", "Sepal width", "Petal length", and "Petal width"

    features    =   prepDataset()    # Collects a dictionary with feature-data sorted by species from the dataset

    for species in features: # Iterate through the three species/classes
        plt.scatter(features[species][xAxis], features[species][yAxis], label=species)    # Add the values to plot according to function arguments

    plt.xlabel(f"{xAxis} [cm]")     # Label the x-axis correctly
    plt.ylabel(f"{yAxis} [cm]")     # Label the y-axis correctly
    plt.legend()                    # Add labels to plot

    #plt.savefig(f"scatterplots/{xAxis}_{yAxis}_scatterplot.png", bbox_inches= "tight") # Saves plot to  an image with correct name
    plt.show()  # Displays the scatterplot

def histograms():

    dataset     =   prepDataset()               # Collects a dictionary with feature-data sorted by species from the dataset
    fig         =   plt.figure(figsize=(8, 5))  # Makes a figure and sets its size. This will contain our subplots

    featureList = { 0:"Sepal length", 1:"Sepal width", 2:"Petal length", 3:"Petal width"}   # Maps indexes to feature labels

    for index in range(4):  # Iterates through the indexes from featureList

        ax = fig.add_subplot(2, 2, index + 1)   # Divides the figure into four subplots and sets axes in the correct place
        ax.set(xlabel='Measurement [cm]', ylabel='Number of samples')  # Labels the axes of the subplot
        fig.tight_layout(pad=.8)  # Sets spacing between the subplots so text labels don't overlap

        for species in dataset:      # Iterates through the species

            featureSamples = dataset[species][featureList[index]]           # Collects the list with samples from the iterated feature
            ax.hist(featureSamples, alpha=0.5, stacked=True, label=species) # Adds the sample values to the histogram

        ax.set_title(featureList[index])    # Set the title of the subplot to the given feature
        ax.legend(prop={'size': 7})         # Displays species labels and sets their size

    #plt.savefig(f"histograms/featuresHistogram.png", bbox_inches= "tight") # Saves histogram to an image file
    plt.show()  # Show the histogram

def plotMSE(trainingDataset, testingDataset, alphas, iterations=1000):

    trainingSampleVectors, trainingRightAnswers     =   makeVectorPairs(trainingDataset)
    testingSampleVectors , testingRightAnswers      =   makeVectorPairs(testingDataset)

     # We need to add an awkward 1 to x_k as described on page 15:
    trainingSampleVectors   =   np.array( [ np.append( sample, [1] ) for sample in trainingSampleVectors ] )
    testingSampleVectors    =   np.array( [ np.append( sample, [1] ) for sample in testingSampleVectors  ] )

    # Get vector representation of label strings
    trainingRightAnswers    =   np.array( [ speciesLabelToVector( label ) for label in trainingRightAnswers ] )
    testingRightAnswers     =   np.array( [ speciesLabelToVector( label ) for label in testingRightAnswers  ] )

    MSEprAlpha = []

    for alpha in alphas:
        _,  MSEprIteration,  _      =   train_linear_classifier(trainingSampleVectors,  trainingRightAnswers,
                                                                testingSampleVectors ,  testingRightAnswers ,
                                                                iterations           ,  alpha               )
        MSEprAlpha.append(MSEprIteration)

    for i in range(len(alphas)):
        MSE = MSEprAlpha[i]
        alpha = alphas[i]

        iteration_numbers = range(len(MSE))
        plt.plot(iteration_numbers, MSE, label='$\\alpha={' + str(alpha) + '}$')

    plt.xlabel("Iteration number")
    plt.ylabel("MSE")
    plt.legend()

    return

