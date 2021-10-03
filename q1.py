# Author - Michael Walia - mpw2217
# Class - 635 Introduction to Machine Learning @ Rochester Institute of Technology
# Date Created - 9-27-2021 1435

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Question 1 - Data Analysis & Visualization

# a - Plotting Raw Features - Scatterplot, Histograms, Line Graphs
class GraphCreation(object):

    def __init__(self):
        self.frogsCSV = np.array([])
        self.frogsSampleCSV = np.array([])


#  I attempted to use np.genfromtxt('Frogs-subsample.csv',delimiter=',',dtype=('unicode'),skip_header=1)
#  however, ran into error 'numpy.ndarray' object has no attribute 'to_numpy'.
#  I then attempted to use pandas and it worked.
    def readFile(self):
        self.frogsCSV = pd.read_csv('Frogs.csv', header=0,
                                          index_col=False)
        self.frogsSampleCSV = pd.read_csv('Frogs-subsample.csv', header=0,
                                          index_col=False)

#  In essence, this is a helper function which allows me to extract MFCCs 10 and 17 from the files.
    def fileManipulation(self, file_name):
        featureMFCC10 = list(file_name)[0]
        featureMFCC17 = list(file_name)[1]
        data = file_name.to_numpy()
        hylaClass = data[data[:, 2] == 'HylaMinuta']
        hypsiboasClass = data[data[:, 2] == 'HypsiboasCinerascens']
        return featureMFCC10, featureMFCC17, hylaClass, hypsiboasClass
#   Draws the 2 scatter plots.
    def drawScatterPlots(self, file_name, name):
        featureMFCC10, featureMFCC17, hylaClass, hypsiboasClass = self.fileManipulation(file_name)

        plt.scatter(hylaClass[:,0], hylaClass[:,1], c='magenta',
                    label="HylaMinuta")
        plt.scatter(hypsiboasClass[:,0], hypsiboasClass[:,1], c='orange',
                    label="HypsiboasCinerascens")
        plt.title('Scatter plot for both classes in ' + name)
        plt.legend(loc='upper right')
        plt.xlabel(featureMFCC10 + ' values')
        plt.ylabel(featureMFCC17 + ' values')
        plt.show()

    #  Here I am drawing 2 histograms for each class [Hyla and Hypsi] (1 per feature) for each frog in each file.
    def drawHistograms(self, file_name, name):
        featureMFCC10, featureMFCC17, hylaClass, hypsiboasClass = self.fileManipulation(file_name)

        plt.hist((hylaClass[:,0],hylaClass[:,1]), color=('magenta','orange'), bins=20,
        label=(featureMFCC10, featureMFCC17))
        plt.title('Histograms of MFCCs for Hyla minuta from ' + name)
        plt.xlabel('Number of Data Points')
        plt.ylabel('Values of Features')
        plt.legend(loc='upper right')
        plt.show()

        plt.hist((hypsiboasClass[:, 0], hypsiboasClass[:, 1]), color=('cyan', 'red'),
                 bins=20, label=(featureMFCC10, featureMFCC17))
        plt.title('Histograms of MFCCs for Hypsiboas cinerascens From '
                  '' + name)
        plt.xlabel('Number of Data Points')
        plt.ylabel('Values of Features')
        plt.legend(loc='upper right')
        plt.show()

    #  Here I am drawing 2 line graphs for each class [Hyla and Hypsi] (1 per feature) for each frog in each file.
    def drawLineGraphs(self, file_name, name):
        featureMFCC10, featureMFCC17, hylaClass, hypsiboasClass = self.fileManipulation(file_name)

        t1 = np.sort(hylaClass[:,0])
        t2 = np.sort(hylaClass[:,1])
        plt.plot(np.arange(t1.shape[0]), t1, color='magenta', label=featureMFCC10)
        plt.plot(np.arange(t2.shape[0]), t2, color='yellow', label=featureMFCC17)
        plt.title('Line graph of MFCCs for Hyla minuta from ' + name)
        plt.xlabel('Number of Data Points')
        plt.ylabel('Values of Features')
        plt.legend(loc='upper right')
        plt.show()

        t1 = np.sort(hypsiboasClass[:, 0])
        t2 = np.sort(hypsiboasClass[:, 1])
        plt.plot(np.arange(t1.shape[0]), t1, color='magenta', label=featureMFCC10)
        plt.plot(np.arange(t2.shape[0]), t2, color='yellow', label=featureMFCC17)
        plt.title('Line Graph of MFCCs for Hypsiboas cinerascens from ' + name)
        plt.xlabel('Number of Data Points')
        plt.ylabel('Values of Features')
        plt.legend(loc='upper right')
        plt.show()

    #  Here I am drawing 1 box plot.
    def drawBoxPlot(self, file_name, name):
        featureMFCC10, featureMFCC17, hylaClass, hypsiboasClass = self.fileManipulation(file_name)

        plt.boxplot(hylaClass[:,:2])
        plt.title('Boxplot for Hyla minuta from ' + name)
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.show()

        plt.boxplot(hypsiboasClass[:, :2])


        plt.title('Box Plot for Hypsiboas cinerascens from ' + name)
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.show()

    #  Here I am drawing 1 box plot.
    def drawErrorBar(self, file_name, name):
        featureMFCC10, featureMFCC17, hylaClass, hypsiboasClass = self.fileManipulation(file_name)

        x_pos = np.arange(4)
        CTEs = [np.mean(hylaClass[:,0]), np.mean(hylaClass[:,1]),
                np.mean(hypsiboasClass[:,0]), np.mean(hypsiboasClass[:,1])]
        error = [np.std(hylaClass[:,0]), np.std(hylaClass[:,1]),
                np.std(hypsiboasClass[:,0]), np.std(hypsiboasClass[:,1])]
        plt.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.6,
                ecolor='black', capsize=10)
        plt.tight_layout()
        plt.title('Bar Graph With Error Bars for' + name)
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.show()

    #1b.
    def calculateStatistics(self, file_name, name):
        featureMFCC10 = list(file_name)[0]
        featureMFCC17 = list(file_name)[1]
        data = file_name.to_numpy()

        featureMFCC10_mean = np.mean(data[:,0])
        featureMFCC17_mean = np.mean(data[:,1])
        hylaClass_STD = np.std(data[:,0])
        hypsiboasClass_STD = np.std(data[:,1])

        temp = data[:,:2].astype(float)
        covarianceMatrix = np.cov(temp[:,:2].T)
        print('Statistics for',name)
        print('Expected Value for', featureMFCC10, ': ', featureMFCC10_mean)
        print('Expected Value for', featureMFCC17, ': ', featureMFCC17_mean)
        print('Covariance Matrix:\n', covarianceMatrix)
        print('STD for', featureMFCC10, ': ', hylaClass_STD)
        print('STD for', featureMFCC17, ': ', hypsiboasClass_STD)
        print()


    def makeGraphs(self):
        print()
        self.readFile()

        self.drawScatterPlots(self.frogsCSV, 'Frogs.csv')
        self.drawScatterPlots(self.frogsSampleCSV, 'Frogs-subsample.csv')
        print('[2/20] Graphs Complete - 2 Scatter Plots for Frogs.CSV and Frogs-subsample.CSV')

        self.drawHistograms(self.frogsCSV, 'Frogs.csv')
        self.drawHistograms(self.frogsSampleCSV, 'Frogs-subsample.csv')
        print('[10/20] Graphs Complete - 8 Histograms for Frogs.CSV and Frogs-subsample.CSV')

        self.drawLineGraphs(self.frogsCSV, 'Frogs.csv')
        self.drawLineGraphs(self.frogsSampleCSV, 'Frogs-subsample.csv')
        print('[18/20] Graphs Complete - 8 Line Graphs for Frogs.CSV and Frogs-subsample.CSV')

        self.drawBoxPlot(self.frogsCSV, 'Frogs.csv')
        self.drawBoxPlot(self.frogsCSV, 'Frogs-subsample.csv')
        print('[19/20] Graphs Complete - 1 Box Plot for Frogs.CSV and Frogs-subsample.CSV')


        self.drawErrorBar(self.frogsCSV, 'Frogs.csv')
        self.drawErrorBar(self.frogsSampleCSV, 'Frogs-subsample.csv')
        print('[20/20] Graphs Complete - 1 Bar Graph With Error Bars for Frogs.CSV and Frogs-subsample.CSV')
        print()

        self.calculateStatistics(self.frogsCSV, 'Frogs.CSV')
        self.calculateStatistics(self.frogsSampleCSV, 'Frogs-subsample.CSV')

#   Creates an object from the GraphCreation class so we can generate the graphs.
if __name__ == '__main__':
    localGraph = GraphCreation()
    localGraph.makeGraphs()