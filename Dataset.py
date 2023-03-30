import numpy as np

# Shuffle dataset does propably not need to be used after all.
def shuffleDataset():  # Reorders the lines from our dataset. This function only needs to be ran once.

    # The given dataset has the first 50 rows be one species, the next 50 another, and so on. Later we will first separate
    # the given data into the first 30 and last 20 sample vectors, then later into the first 20 and last 30. We will
    # use these sample vectors to train our classifier. The problem with this is that if we separate out the first 30
    # sample vectors from the dataset as it is in iris.data.txt, our classifier will only train on data from Iris Setosa.
    # Then it will run tests only on Iris Virginica. Obviously this is not ideal. This funtion therefore shuffles the
    # lines in the original dataset so that the sequence of lines will be Setosa, Versicolor, Virginica. It then writes
    # the new lines to shuffledDataset.txt, which is the file we will be using for the rest of the project.
    # Our classifier then train and test on a close to equal amount of data from each species.

    with open('iris.data.txt', 'r') as dataset:     # Open the original dataset

        lines = dataset.readlines()    # Make a list containing every line in the dataset as is

        orderedDataset    =   []      # This list will contain the new order of lines

        index       =   0           # Starting index
        for i in range(50):         # Run this sequence 50 times since there are 50 data entries pr. species
            orderedDataset.append(lines[index])
            orderedDataset.append(lines[index+50])    # Add the first, second, thrid... data entry of each species
            orderedDataset.append(lines[index+100])
            index += 1  # Update the index so the next iteration wil add the next data entry for every species

    with open('shuffledDataset.txt', 'w') as newDataset:    # Open a new file we can write to

        newDataset.writelines(orderedDataset)     # Write the new lines to shuffledDataset.txt

def prepDataset():

# This function makes arrays with sample values and species and will return the following:
    # prepDataset[0]    = all Iris Setosa features in order
    # prepDataset[1]    = all Iris Versicolour features in order
    # prepDataset[2]]   = all Iris Virginica features in order

    # The feature of each species is given by the index prepDataset[species][feature]:
        # [0]   =   sepal length
        # [1]   =   sepal width
        # [2]   =   petal length
        # [3]   =   petal width

    with open('iris.data.txt') as dataset:      # Open the dataset
        lines = dataset.readlines()             # Makes a list containing every line in the dataset

    data_labels = []     # Makes a list containing data/label pairs: [(data, label), (data, label), ...]

    for line in lines:
        cells = line.split(",")     # Iterates throug every line and create a list with each value/label

        data = [float(cells[i]) for i in range(0, 4)]   # Makes list with all sample-data
        data = np.array(data, dtype=np.float32)         # Convert sample-data list to numpy array

        label = cells[4][:len(cells[4]) - 1]    # Makes list with species label and removes \n from the end
        data_labels.append([data, label])       # Adds the sample-data/label-pair to data_labels list as a tuple

    data, labels = np.transpose(data_labels)        # make two lists: one with sample-data arrays and one with just species labels
    data, labels = np.array(data), np.array(labels) # Create numpy arrays with the above

    # Now that we have separated the species and the sample-data, we want to sort it by features.

    classes             =   np.unique(labels)       # Makes a list with the classes given in the dataset [species1, species2, species3]
    featuresBySpecies   =   {}                      # Makes a dictionary that will contain the species and then another dictionary with the features

    for species in classes:  # Iterate through the species/classes

        indexes = np.where(labels == species)[0]  # Finds the indexes where each species has been shuffled to

        sampleData = {  # Creates a dictionary with keywords = the function arguments
            "Sepal length"  : [data[i][0] for i in indexes],
            "Sepal width"   : [data[i][1] for i in indexes],
            "Petal length"  : [data[i][2] for i in indexes],
            "Petal width"   : [data[i][3] for i in indexes]
        }

        featuresBySpecies[species] = sampleData     # Adds a dictionary with all current species features to featuresBySpecies

    return featuresBySpecies    # Returns the full dataset

def split_dataset(dataset, splitIndex):

    # This function will separate the input data (from prepDataset()) into a training- and a testing set. The first splitIndex numbers of
    # data entries pr. species will be collected in a dictionary called firstSet, and the last entries in the
    # dictionary lastSet

    firstSet, lastSet   =   {},     {}      # Create the dictionaries

    for species in dataset:     # Iterate through the species

        firstSet[species]   =   {}      # Create another dictionary inside the dictionaries. These will contain
        lastSet[species]    =   {}      # the four different features

        for feature in dataset[species]:    # Iterate through the different features

            # Add the splitIndex number first feature samples to their corresponding dictionary
            firstSet[species][feature]  =   dataset[species][feature][:splitIndex]
            lastSet[species][feature]   =   dataset[species][feature][splitIndex:]

    return firstSet, lastSet    # Return the two sets

def remove_feature(dataset, featureName): # Removes chosen feature from every species in the dataset

    for species in dataset:     # Iterate through the species
        del dataset[species][featureName]   # For every species, delete the chosen keyword and its values

    return dataset  # Return the modified dataset

def makeSampleVectors(dataset):

    sampleVectors   =   []  # Make a list for the sample vectors
    rightAnswers    =   []  # Make a list for the correct label vectors corresponding to the sample vectors

    for species in dataset:  # Iterate through the species

        vector  =   []  # Temporary list to hold the values pr. feature for the current iterated species

        loops   =   0   # Checks how many times the for loop under has ran
        for feature in dataset[species]:  # Iterate through the features which is four lists of feature data

            vector.append(dataset[species][feature])  # Append these four lists to the temporary list

            if loops == 0:  # Run the speciesToLabel() function and add output to labels[], but only the first loop

                # Appends as many of one species vector as there are sample vectors for the current species
                for i in range(len(dataset[species][feature])):
                    rightAnswers.append(speciesLabelToVector(species))



            loops   +=  1   # Increment the loop counter by one

        # The vector list is now in the form [ [f1, f1, f1, f1, ...], [f2, f2, f2, f2, f2, ...], ... ]
        # We want it on the form [[f1, f2, f3, f4], ...] so that each vector contains data from all the features
        # of one specific Iris sample

        vector = np.transpose(vector)   # Transpose so that we get an array of vectors like [[f1, f2, f3, f4], ...]
        sampleVectors.extend(vector)    # Add the sample vectors from the current species to the sampleVectors list

        # For each sample vector we make a list with its species label corresponding vector representation. The index
        # of each of these label vector will be the same as its corresponding sample vector in sampleVectors[].

    sampleVectors   =   np.array(sampleVectors)  # Convert both lists to numpy arrays
    rightAnswers    =   np.array(rightAnswers)

    return sampleVectors, rightAnswers      # Return an array with the sample vectors and an array with the correct species

def speciesLabelToVector(speciesLabel):

    classes     =   np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])    # Creates list with species name in order

    index = int( np.where(classes == speciesLabel)[0] )  # Finds the index where the species match

    # Makes a new vector with 1 at the correct index, and 0 at the others
    speciesVector = np.array([ i == index for i in range(3) ], dtype=np.int)

    return speciesVector    # returns the vector value corresponding to the species name

def speciesVectorToLabel(speciesVector):

    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  # Creates list with species name in order

    speciesIndex    =   np.argmax(speciesVector)    # Finds the index with the highest value

    speciesLabel    =   classes[speciesIndex]       # Picks the correct species from classes

    return speciesLabel     # Returns the species name as a string

