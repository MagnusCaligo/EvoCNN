import cv2
import numpy as np
import random
import copy

class Config:
    def __init__(self, configFilePath):
        configFile = open(configFilePath, "r")
        line = configFile.readline()

        self.configParameters = {}

        while line:
            line = line.replace("\n", "")
            line = line.split("=")
            if len(line) > 0:
                if line[0] != "" and line[0][0] == "#":
                    line = configFile.readline()
                    continue
                if len(line) >=2 :
                    self.configParameters[line[0].replace(" ","")] = line[1].replace(" ","")
            line = configFile.readline()
    def __getitem__(self, item):
        return self.configParameters[item]


def initalizePopulation(config):
    maxConvPooling = int(config["maxConvPooling"])
    maxFCLayers = int(config["maxFC"])
    popSize = int(config["populationSize"])

    population = []
    while len(population) < popSize:
        #initalize Part1 i.e. convolutional and pooling layers
        part1 = []
        randVal = random.randint(1, maxConvPooling)
        while len(part1) < randVal:
            r = random.uniform(0,1)
            layer = None
            if r <= 0.5:
                layer = _initalizeConvolutionalLayer(config)
            else:
                layer = _initalizePoolingLayer(config)
            part1.append(layer)

        #initalize fully connected layers
        part2 = []
        randVal = random.randint(1, maxFCLayers)
        while len(part2) < randVal:
            part2.append(_initalizeFCLayer(config))

        population.append(part1 + part2)
    return population


def _initalizeConvolutionalLayer(config):
    layer = {}
    layer["type"] = "conv"
    layer["operation"] = random.choice(config["convolutionTypes"].split(","))
    layer["width"] = random.randint(1, int(config["maxFilterWidth"]))
    layer["height"] = random.randint(1, int(config["maxFilterWidth"]))
    layer["numFeatureMaps"] = random.randint(1,int(config["maxFeatureMaps"]))
    layer["strideWidth"] = random.randint(1, int(config["maxStrideWidth"]))
    layer["strideHeight"] = random.randint(1, int(config["maxStrideHeight"]))
    layer["standardDeviation"] = random.uniform(0, int(config["maxFilterStandardDeviation"]))
    layer["mean"] = random.uniform(0, int(config["maxFilterMean"]))
    return layer

def _initalizePoolingLayer(config):
    layer = {}
    layer["type"] = "pool"
    layer["operation"] = random.choice(config["poolingTypes"].split(","))
    layer["width"] = random.randint(1, int(config["maxFilterWidth"]))
    layer["height"] = random.randint(1, int(config["maxFilterWidth"]))
    layer["strideWidth"] = random.randint(1, int(config["maxStrideWidth"]))
    layer["strideHeight"] = random.randint(1, int(config["maxStrideHeight"]))
    layer["standardDeviation"] = random.uniform(0, int(config["maxFilterStandardDeviation"]))
    layer["mean"] = random.uniform(0, int(config["maxFilterMean"]))
    return layer

def _initalizeFCLayer(config):
    layer = {}
    layer["type"] = "fullyConnected"
    layer["numOfNeurons"] = random.randint(1, int(config["maxNeuronsFC"]))
    layer["standardDeviation"] = random.uniform(0, int(config["maxNeuronStandardDeviation"]))
    layer["mean"] = random.uniform(0, int(config["maxNeuronMean"]))
    return layer


#used for testing implementation of network architecture; should not be used anywhere else
def runGeneomeOnImage(geneome, img):
    imgTemp1 = copy.copy(img)
    imgTemp2 = copy.copy(img)
    for index, layer in enumerate(geneome):
        if layer['type'] == "conv":
            pass
        elif layer['type'] == 'pool':
            pass
        elif layer['type'] == 'fullyConnected':
            pass


def convolve(filterMap, kernelSize, kernel, stride, numFilters):
    height, width, channels = filterMap.shape
    #Find center of kernelSize
    kX = (kernelSize[0] / 2) + 1
    kY = (kernelSize[1] / 2) + 1

    #Check if kernelSize has an even side
    if kernelSize[0] % 2 == 0:
        kX = kernelSize[0] /2
    if kernelSize[1] % 2 == 0:
        kY = kernelSize[1] /2

    outputWidth = width - ((kernelSize[0]))
    outputHeight = height - ((kernelSize[1]))
    outputFilters = np.zeros((outputHeight, outputWidth, numFilters), dtype="float32")

    print "input size", filterMap.shape
    print "output size", outputFilters.shape


    for y in np.arange(0, outputHeight):
        for x in np.arange(0, outputWidth):
            roiy1 = y
            roiy2 = y + (kernelSize[1])
            roix1 = x
            roix2 = x + (kernelSize[0])
            roi = filterMap[roiy1:roiy2, roix1:roix2, 0]
            print "roi size", roi.shape, y, x, roix1, roix2, outputWidth
            k = (roi * kernel).sum()
            #outputFilters[y,x,0] = filterMap[y,x,1]
            outputFilters[y,x,0] = k

    #output = rescale_intensity(output, in_range(0,255))
    outputFilters = (outputFilters * 255).astype("uint8")
    return outputFilters
            
        

    


