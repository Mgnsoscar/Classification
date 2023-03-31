import numpy as np

def makeVectorPairs(splitIndex=49):

    # Reads the dataset and makes sample-/class vector pairs. Makes two sets of these according to the chosen splitIndex.

    classes = np.array( ["Iris-setosa", "Iris-versicolour", "Iris-virginica"] )

    firstSetSampleVectors   =   []
    firstSetClassVectors    =   []
    lastSetSampleVectors    =   []
    lastSetClassVectors     =   []

    for species in classes:
        with open(f"C:/Users/mgnso/PycharmProjects/Classification/classification2/datasets/{species}.txt") as sampleData:

            tempSampleVectors   =   []
            tempClassVectors    =   []

            lines = sampleData.readlines()
            for line in lines:

            # Make each sample vector:
                cells               =   line.split(",")     # Split line into list of values
                cells[len(cells)-1] =   cells[len(cells)-1][:-1]    # Reomve "\n" from end of last value

                cells.append(1)     # Add the 1 at the end of each samplevector as described in report

                sampleVector        =   np.array(cells, dtype=float)    # Make a numpy array of the values
                tempSampleVectors.append(sampleVector)

            # Make each class vector:
                index = int(np.where(classes == species)[0])  # Finds the index where the species match

                # Makes a new vector with 1 at the correct index, and 0 at the others
                classVector     =   []
                classVector.append([int(i == index) for i in range(3)])

                classVector     =   np.array(classVector)
                tempClassVectors.extend(classVector)

            firstSetSampleVectors.extend(tempSampleVectors[:splitIndex])
            firstSetClassVectors.extend(tempClassVectors[:splitIndex])
            lastSetSampleVectors.extend(tempSampleVectors[splitIndex:])
            lastSetClassVectors.extend(tempClassVectors[splitIndex:])

    firstSetClassVectors, firstSetSampleVectors = np.array(firstSetClassVectors), np.array(firstSetSampleVectors)
    lastSetClassVectors,  lastSetSampleVectors  = np.array(lastSetClassVectors) , np.array(lastSetSampleVectors)

    firstSet = [firstSetSampleVectors, firstSetClassVectors]
    lastSet  = [lastSetSampleVectors , lastSetClassVectors ]

    return firstSet, lastSet

def remove_feature(sampleVectors, featureName): # Removes chosen feature from every species in the dataset

    features = {"Sepal length":0, "Sepal width":1, "Petal length":2, "Petal width":3}

    for vector in sampleVectors:    # Iterate through each sample/class vector pair.
        vector[0]  = np.delete(vector[0], features[featureName])    # Delete the index corresponing to chosen feature

    return sampleVectors  # Return the modified dataset

def speciesVectorToLabel(speciesVector):

    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # Creates list with species name in order

    speciesIndex    =   np.argmax(speciesVector)    # Finds the index with the highest value

    speciesLabel    =   classes[speciesIndex]       # Picks the correct species from classes

    return speciesLabel     # Returns the species name as a string


first, last = makeVectorPairs(30)

