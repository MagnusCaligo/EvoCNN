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
            if r <= float(config["chanceOfConvolutionalLayer"]):
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
    sd = []
    mean = []
    for i in range(int(layer["numFeatureMaps"])):
        sd.append(random.uniform(0, int(config["maxFilterStandardDeviation"])))
        mean.append(random.uniform(0, int(config["maxFilterMean"])))
    layer["standardDeviation"] = sd
    layer["mean"] = mean
    return layer

def _initalizePoolingLayer(config):
    layer = {}
    layer["type"] = "pool"
    layer["operation"] = random.choice(config["poolingTypes"].split(","))
    layer["width"] = random.randint(1, int(config["maxFilterWidth"]))
    layer["height"] = random.randint(1, int(config["maxFilterWidth"]))
    layer["strideWidth"] = random.randint(1, int(config["maxStrideWidth"]))
    layer["strideHeight"] = random.randint(1, int(config["maxStrideHeight"]))
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
    imgTemp1 = copy.deepcopy(img)
    imgTemp2 = copy.deepcopy(img)
    for index, layer in enumerate(geneome):
        print "Layer Type:", layer["type"],
        height, width, channels = imgTemp1.shape
        if layer['type'] == "conv":
            kernelSize = (int(layer["width"]), int(layer["height"]))
            stride = (int(layer["strideWidth"]), int(layer["strideHeight"]))
            outputWidth = int((width - ((kernelSize[0])))/stride[0])
            outputHeight = int((height - ((kernelSize[1])))/stride[1])
            print "Kernel Size:", kernelSize, "Stride:", stride, "Feature Maps:", layer["numFeatureMaps"]
            if outputWidth < 0:
                outputWidth = kernelSize[0]
            if outputHeight < 0:
                outputHeight = kernelSize[1]
            imgTemp2 = np.zeros((outputHeight, outputWidth, int(layer["numFeatureMaps"])))

            for i in range(int(layer["numFeatureMaps"])):
                kernel = np.random.normal(float(layer["mean"][i]), float(layer["standardDeviation"][i]), kernelSize)
                featureMap = convolve(imgTemp1, kernelSize, kernel, stride, layer["operation"])
                #print "Final Shape:", imgTemp2.shape, outputWidth, outputHeight, kernelSize, stride
                #print "Final:", width, height
                #print "Final Kernel:", kernel
                imgTemp2[:,:,i] = featureMap

            imgTemp1 = copy.deepcopy(imgTemp2)

        elif layer['type'] == 'pool':
            kernelSize = (int(layer["width"]), int(layer["height"]))
            stride = (int(layer["strideWidth"]), int(layer["strideHeight"]))
            outputWidth = int((width - ((kernelSize[0])))/stride[0])
            outputHeight = int((height - ((kernelSize[1])))/stride[1])
            print "Kernel Size:", kernelSize, "Stride:", stride, "Operation:", layer["operation"]
            if outputWidth <= 0:
                outputWidth = kernelSize[0]
            if outputHeight <= 0:
                outputHeight = kernelSize[1]
            imgTemp2 = np.zeros((outputHeight, outputWidth, channels))
            for i in range(channels):
                kernel = np.array([])
                featureMap = convolve(imgTemp1[:,:,i], kernelSize, kernel, stride, layer["operation"])
                imgTemp2[:,:,i] = featureMap
            imgTemp1 = copy.deepcopy(imgTemp2)

        elif layer['type'] == 'fullyConnected':
            imgTemp1 = copy.copy(imgTemp2)
            numNeurons = layer["numNeurons"]
            standardDeviation = layer["standardDeviation"]
            mean = layer["mean"]
            print "Number of Neurons:", numNeurons

    return imgTemp1


def convolve(filterMap, kernelSize, srcKernel, stride, convolutionType):
    kernel = None
    if convolutionType not in ["average", "max", "min"]:
        height, width, channels = filterMap.shape
        kernel = np.transpose(srcKernel)
        kernel = np.repeat(kernel[:,:, np.newaxis], channels, axis=2)
    else:
        filterMap = np.repeat(filterMap[:,:, np.newaxis], 1, axis=2)
        height, width, channels = filterMap.shape
        channels = 1

    outputWidth = int((width - ((kernelSize[0])))/stride[0])
    outputHeight = int((height - ((kernelSize[1])))/stride[1])
    if outputWidth <= 0:
        outputWidth = kernelSize[0]
    if outputHeight <= 0:
        outputHeight = kernelSize[1]
    outputFilters = np.zeros((outputHeight, outputWidth), dtype="float32")
    #print "Output Shape:", outputFilters.shape, outputWidth, outputHeight, kernelSize, stride
    #print "Output:", width, height
    #print "Convolve Kernel:", kernel[:,:,0]

    for y in np.arange(0, outputHeight):
        for x in np.arange(0, outputWidth):
            roiy1 = (y * stride[1])
            roiy2 = (y * stride[1])  + (kernelSize[1])
            roix1 = (x * stride[0])
            roix2 = (x * stride[0]) + (kernelSize[0])
            roi = filterMap[roiy1:roiy2, roix1:roix2, 0:channels]
            if 0 in roi.shape:
                continue
            '''print roi.shape, kernel.shape
            print kernelSize
            print roiy1, roiy2
            '''
            if convolutionType == "sum":
                k = (roi * kernel).sum()
            elif convolutionType == "mean":
                k = (roi * kernel).mean()
            elif convolutionType == "average":
                k = roi.sum()/roi.size
            elif convolutionType == "max":
                #print "Maxing:", roi.shape, kernelSize, roix1, roix2, filterMap.shape
                k = np.amax(roi)
            elif convolutionType == "min":
                #print "Minning:", roi.shape, kernelSize, roix1, roix2, filterMap.shape
                k = np.amin(roi)
            outputFilters[y,x] = k

    #outputFilters = (outputFilters * 255).astype("uint8")
    return outputFilters
            
        

    


