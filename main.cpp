#include <vector>
#include <iostream>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <iomanip>

//made from https://www.youtube.com/watch?v=KkwX7FkLfug
//TODO: add training data
//TODO: debug and run

using namespace std;

struct Connection {
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const {return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    vector<Connection> m_outputWeights;
private:
    constexpr static double eta = 0.2;
    constexpr static double alpha = 0.5;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal, m_gradient;
    unsigned m_myIndex;
};

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (unsigned n = 0; n < prevLayer.size() - 1; n++) {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x) {
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1 - x*x;
}

void Neuron::feedForward(const Layer &prevLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); n++) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    m_myIndex = myIndex;
    for (unsigned c = 0; c < numOutputs; c++) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
}


class Net {
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
private:
    double m_error;
    double m_recentAverageError;
    const double m_recentAverageSmoothingFactor = 0.1;
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};


void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear();
    for (unsigned n = 0; n < m_layers.back().size() - 1; n++) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
};


void Net::backProp(const vector<double> &targetVals) {
    // calculate overall net error
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);
    
    // implement a recent average mesurement
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);
    
    //cout <<"Error: " << setprecision(2) << fixed << m_recentAverageError << endl;
    
    // calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }
    
    // calculate gradients in hidden layers
    for (long layerNum = m_layers.size() - 2; layerNum > 0; layerNum--) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];
        for (unsigned n = 0; n < hiddenLayer.size(); n++) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }
    
    // for all layers from outputs to first hidden layer,
    // update connection weights
    for (long layerNum = m_layers.size() - 1; layerNum > 0; layerNum--) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < layer.size() - 1; n++) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const vector<double> &inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1);
    for (unsigned i = 0; i < inputVals.size(); i++) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology) {
    long numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a neuron " << endl;
        }
        m_layers.back().back().setOutputVal(1.0);
    }
}

void makeTest(Net &myNet) {
    vector<double> inputVals, resultVals;
    for (unsigned i = 0; i < 2; i++) {
        for (unsigned j = 0; j < 2; j++) {
            inputVals.clear();
            inputVals.push_back(i);
            inputVals.push_back(j);
            myNet.feedForward(inputVals);
            myNet.getResults(resultVals);
            cout << i << " " << j << "->" << resultVals[0] << endl;
        }
    }
}


int main(int argc, const char * argv[]) {
    // e.g. {2, 2, 1}
    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(8);
    topology.push_back(4);
    topology.push_back(1);
    Net myNet(topology);
    vector<double> inputVals, targetVals, resultVals;
    
    cout << "Before..." << endl;
    makeTest(myNet);
    

    for (long i = 0; i < 150000; i++) {
        inputVals.clear();
        targetVals.clear();
        double sum = 1;
        for (unsigned j = 0; j < 2; j++) {
            inputVals.push_back(rand() % 2);
            sum = sum && inputVals.back();
        }
        targetVals.push_back(sum);
                
        myNet.feedForward(inputVals);
        myNet.backProp(targetVals);
    }
    
    cout << "After..." << endl;
    makeTest(myNet);
    
    exit(3);
    return 0;
}


