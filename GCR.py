# -*- coding: utf-8 -*-
from PIL import Image
from numpy import array
import sqlite3
import tkMessageBox

import matplotlib.pyplot as plt

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import SigmoidLayer

# global db, x, dimage, image,alphabet

alphabet = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
            10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's',
            19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}


class Sample:
    def __init__(self, Input, Target, Id=None):
        self.Id = Id
        self.Input = Input
        self.Target = Target

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.Input == other.Input and self.Target == other.Target

    def __ne__(self, other):
        return not self.__eq__(other)

    def getInput(self):
        inp = self.Input.split(',')
        return [int(i) for i in inp]

    def getTarget(self):
        tar = self.Target.split(',')
        print tar
        return [int(i) for i in tar]


class Params:
    def __init__(self, Weights, ID=None):
        self.ID = ID
        self.Weights = Weights

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.Weights == other.Weights

    def __ne__(self, other):
        return not self.__eq__(other)

    def getWeights(self):
        w = self.Weights.split(',')
        return [float(i) for i in w]


def getcharkey(char):
    for key, ch in alphabet.iteritems():
        if ch.decode('utf-8') == char:
            return key


def init():
    global samples, db

    # caching samples

    samples = []
    db = sqlite3.connect('data.db')
    cursor = db.cursor()
    rows = cursor.execute('SELECT *FROM samples')
    rows = rows.fetchall()

    for r in rows:
        sample = Sample(r[1], r[2])
        samples.append(sample)

    global net, ds, trainer

    ins = 256
    hids = ins * 2 / 3
    outs = 26

    net = buildNetwork(ins, hids, outs, bias=True, outclass=SoftmaxLayer)
    ds = SupervisedDataSet(ins, outs)

    rows = cursor.execute('SELECT * FROM parameters')
    rows = rows.fetchall()

    params_list = []

    for r in rows:
        params = Params(r[1])
        params_list.append(params)

    if len(params_list) != 0:
        params = params_list[len(params_list) - 1]
        net._setParameters(params.getWeights())
    trainer = BackpropTrainer(net, ds)

    if len(samples) > 0:

        for s in samples:
            ds.addSample(s.getInput(), s.getTarget())


def which(dim):
    dim = makelist(dim)
    # print dim
    out = net.activate(dim)
    index = out.argmax()
    print alphabet[index]
    print str(out[index] * 100) + '%'
    # print [i for i in out]

    plt.clf()
    plt.title("Graph")
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
              'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    x = range(26)
    plt.xticks(x, labels)
    plt.bar(x, out)

    plt.show()


def train():
    error = 10
    it = 0
    iterations = []
    errors = []
    while error > 0.00001:
        error = trainer.train()
        it += 1
        print "Iteration: " + str(it), "Error: " + str(error)
        iterations.append(it)
        errors.append(error)

    params = makestring(net.params)
    cursor = db.cursor()
    cursor.execute("INSERT INTO parameters (Weights) VALUES (?)", (params,))
    db.commit()

    plt.clf()
    plt.xlabel("Iterations")
    plt.ylabel("Errors")
    plt.title("Error Graph")
    plt.plot(iterations, errors)
    plt.show()

    print 'training finished'


def close():
    db.close()


def average(numlist):
    return sum(numlist) / len(numlist)


def blackwhite(dim):
    dim = dim.tolist()
    imrow = []
    im = []
    for i in dim:
        for j in i:
            imrow.append(average(j))
        im.append(imrow)
        imrow = []

    dim = array(im)
    return dim


def makestring(dim):
    string = [str(i) for i in dim]
    string = ','.join(string)

    return string


def makelist(dim):
    lst = []
    for i in dim:
        for j in i:
            lst.append(j)

    return lst


def addSample(sample):
    samples.append(sample)
    ds.addSample(sample.getInput(), sample.getTarget())
    cursor = db.cursor()
    cursor.execute("INSERT INTO samples (Input,Target) VALUES (?,?)", [sample.Input, sample.Target])
    db.commit()


def getUpRow(dimage):
    x = dimage.shape[0]
    y = dimage.shape[1]

    for i in range(x):
        for j in range(y):
            if average(dimage[i][j]) < 255:
                return i


def getLeftCol(dimage):
    x = dimage.shape[0]
    y = dimage.shape[1]

    for j in range(y):
        for i in range(x):
            if average(dimage[i][j]) < 255:
                return j


def getDownRow(dimage):
    x = dimage.shape[0]
    y = dimage.shape[1]

    for i in range(x - 1, -1, -1):
        for j in range(y - 1, -1, -1):
            if average(dimage[i][j]) < 255:
                return i


def getRightCol(dimage):
    x = dimage.shape[0]
    y = dimage.shape[1]

    for j in range(y - 1, -1, -1):
        for i in range(x - 1, -1, -1):
            if average(dimage[i][j]) < 255:
                return j


def getBox(dimage):
    rowUp = getUpRow(dimage)
    colLeft = getLeftCol(dimage)
    rowDown = getDownRow(dimage)
    colRight = getRightCol(dimage)

    return (colLeft, rowUp, colRight, rowDown)


init()
