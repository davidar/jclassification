/*
* Copyright (C) 2010-2011 David A Roberts <d@vidr.cc>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

package cc.vidr.jclassification.ann;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * Implements a multi-layer feed-forward artificial neural network.
 * Training is performed with stochastic back-propagation.
 * 
 * @author  David A Roberts
 */
public class ANN implements Serializable {
    private static final long serialVersionUID = -7509606549734140402L;
    private static final Random random = new Random();
    
    public final int NUM_LAYERS, INPUT_LAYER, FIRST_HIDDEN_LAYER,
                     LAST_HIDDEN_LAYER, OUTPUT_LAYER;
    
    /** Connection weights [layer of j][j][i] */
    private double[][][] w;
    /** Current unit activations [layer][j] */
    private double[][] y;
    /** Current delta values [layer][j] */
    private double[][] delta;
    
    /**
     * Create a new ANN.
     * 
     * @param layerSizes  the number of units in each layer, beginning with
     *                    the input layer, followed by the hidden layer(s),
     *                    and finally the output layer
     */
    public ANN(int... layerSizes) {
        NUM_LAYERS = layerSizes.length;
        INPUT_LAYER = 0;
        FIRST_HIDDEN_LAYER = 1;
        LAST_HIDDEN_LAYER = NUM_LAYERS - 2;
        OUTPUT_LAYER = NUM_LAYERS - 1;
        this.w = new double[NUM_LAYERS][][];
        this.y = new double[NUM_LAYERS][];
        this.delta = new double[NUM_LAYERS][];
        
        for(int layer = INPUT_LAYER; layer <= OUTPUT_LAYER; layer++) {
            int layerSize = layerSizes[layer];
            if(layer != OUTPUT_LAYER)
                layerSize++; // bias unit
            
            // create activation and delta arrays for this layer
            this.y[layer] = new double[layerSize];
            this.delta[layer] = new double[layerSize];
            if(layer != OUTPUT_LAYER)
                this.y[layer][0] = 1; // bias unit
            
            if(layer != INPUT_LAYER) {
                // initialise weights to small random values
                int prevLayerSize = layerSizes[layer-1] + 1;
                this.w[layer] = new double[layerSize][prevLayerSize];
                for(int j = nonBiasUnit(layer); j < size(layer); j++)
                    for(int i = 0; i < size(layer-1); i++)
                        this.w[layer][j][i] = random.nextGaussian()*0.1;
            }
        }
    }
    
    /**
     * Update the activations of all units in the network.
     * 
     * @param input  the input vector
     * @return       the output vector
     */
    public double[] feedForward(double[] input) {
        // set activations of input units
        for(int i = nonBiasUnit(INPUT_LAYER); i < size(INPUT_LAYER); i++)
            y[INPUT_LAYER][i] = input[i-1];
        
        // calculate activations of units in all following layers
        for(int layer = FIRST_HIDDEN_LAYER; layer <= OUTPUT_LAYER; layer++) {
            for(int j = nonBiasUnit(layer); j < size(layer); j++) {
                double x = 0;
                for(int i = 0; i < size(layer-1); i++)
                    x += w[layer][j][i] * y[layer-1][i];
                y[layer][j] = sigmoid(x);
            }
        }
        return y[OUTPUT_LAYER];
    }
    
    /**
     * Perform a single iteration of back-propagation.
     * 
     * @param d    the target vector
     * @param eta  the learning rate
     */
    public void backProp(double[] d, double eta) {
        // calculate error of output units
        for(int k = 0; k < size(OUTPUT_LAYER); k++) {
            double y_k = y[OUTPUT_LAYER][k];
            delta[OUTPUT_LAYER][k] = (y_k - d[k]) * sigmoidDerivative(y_k);
        }
        
        // back-propagate error to hidden units
        for(int layer  = LAST_HIDDEN_LAYER;
                layer >= FIRST_HIDDEN_LAYER; layer--) {
            for(int j = nonBiasUnit(layer); j < size(layer); j++) {
                double sum = 0;
                for(int k = nonBiasUnit(layer+1); k < size(layer+1); k++)
                    sum += w[layer+1][k][j] * delta[layer+1][k];
                delta[layer][j] = sigmoidDerivative(y[layer][j]) * sum;
            }
        }
        
        // perform gradient descent
        for(int layer = FIRST_HIDDEN_LAYER; layer <= OUTPUT_LAYER; layer++) {
            for(int j = nonBiasUnit(layer); j < size(layer); j++) {
                for(int i = 0; i < size(layer-1); i++) {
                    w[layer][j][i] -= eta * delta[layer][j] * y[layer-1][i];
                }
            }
        }
    }
    
    /**
     * Train the network with stochastic back-propagation.
     * 
     * @param inputs   a list of input vectors
     * @param outputs  a list of corresponding output vectors
     * @param n        the number of training iterations to perform
     * @param eta      the learning rate
     */
    public void train(double[][] inputs, double[][] outputs,
            int n, double eta) {
        for(int i = 0; i < n; i++) {
            // choose a random sample
            int sample = random.nextInt(inputs.length);
            double[] input = inputs[sample], output = outputs[sample];
            // calculate activations of all units
            feedForward(input);
            // update weights via back-propagation
            backProp(output, eta);
        }
    }
    
    /**
     * Get the activations of the non-bias units in the given layer.
     * 
     * @param layer  the layer to retrieve
     * @return       the activations
     */
    public double[] getActivations(int layer) {
        return Arrays.copyOfRange(y[layer], nonBiasUnit(layer), size(layer));
    }
    
    /**
     * Get the weights of the incoming connections to the given unit.
     * 
     * @param layer  the layer the unit is in
     * @param j      the index of the unit
     * @return       the array of weights
     */
    public double[] getWeights(int layer, int j) {
        return w[layer][j];
    }
    
    /**
     * The sigmoid function.
     * 
     * @param x  the input value (x)
     * @return   the output value (y)
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
    
    /**
     * The derivative of the sigmoid function.
     * 
     * @param y  the output value (y)
     * @return   the gradient with respect to x
     */
    private double sigmoidDerivative(double y) {
        return y * (1.0 - y);
    }
    
    /**
     * Return the number of units in the given layer.
     */
    private int size(int layer) {
        return y[layer].length;
    }
    
    /**
     * Returns the index of the first non-bias unit in the given layer.
     * This is 0 for the output layer, and 1 for all other layers.
     */
    private int nonBiasUnit(int layer) {
        return layer == OUTPUT_LAYER ? 0 : 1;
    }
    
    /**
     * Train a 2:2:1 network to learn the XOR function.
     */
    public static void main(String[] args) {
        // create a new network with 2 input units,
        // a single hidden layer containing 2 units,
        // and 1 output unit
        ANN nn = new ANN(2, 2, 1);
        // an extra hidden layer of k units could be created by
        //    new ANN(2, 2, k, 1)
        // an arbitrary number of layers may be specified
        
        // specify the input-output pairs for the XOR function
        double[][] inputs = {{0,0},{0,1},{1,0},{1,1}},
                  outputs = {  {0},  {1},  {1},  {0}};
        // print the outputs of the untrained network
        nn.printOutputs(inputs);
        // train with n=10^6,eta=0.1
        nn.train(inputs, outputs, 1000000, 0.1);
        // print the outputs of the trained network
        nn.printOutputs(inputs);
    }

    private void printOutputs(double[][] inputs) {
        for(double[] input : inputs) {
            // calculate the network's output for this input
            double[] output = feedForward(input);
            System.out.println(
                    Arrays.toString(input) + " : " +
                    Arrays.toString(output));
        }
        System.out.println();
    }
}
