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

package cc.vidr.jclassification.svm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import cc.vidr.jclassification.svm.kernel.GaussianKernel;
import cc.vidr.jclassification.svm.kernel.Kernel;
import cc.vidr.jclassification.svm.kernel.LinearKernel;
import cc.vidr.jclassification.svm.vector.DataVector;
import cc.vidr.jclassification.svm.vector.RealVector;

/**
 * An implementation of a Support Vector Machine, with support for
 * soft-margin and arbitrary kernel functions.
 * 
 * @author  David A Roberts
 */
public class SVM implements Serializable {
    private static final long serialVersionUID = 7667970753574953210L;
    public static final double EPSILON = 1e-3;
    /** The support vectors */
    List<SupportVector> vectors = new ArrayList<SupportVector>();
    /** The threshold */
    double b = 0;
    /** The kernel function */
    Kernel kernel;
    /** The soft-margin parameter */
    double c;
    
    /**
     * Create a soft-margin SVM.
     * 
     * @param kernel  the kernel function
     * @param c       the soft-margin parameter
     */
    public SVM(Kernel kernel, double c) {
        this.kernel = kernel;
        this.c = c;
    }
    
    /**
     * Create a hard-margin SVM.
     * 
     * @param kernel  the kernel function
     */
    public SVM(Kernel kernel) {
        this(kernel, Double.POSITIVE_INFINITY);
    }
    
    /**
     * Create a hard-margin SVM with a linear kernel.
     */
    public SVM() {
        this(new LinearKernel());
    }
    
    /**
     * Add the given example to the SVM.
     * 
     * @param x  the input vector
     * @param y  the target class
     */
    public void add(DataVector x, int y) {
        vectors.add(new SupportVector(x, y));
    }
    
    /**
     * Throw away all non-support vectors.
     */
    public void prune() {
        Iterator<SupportVector> iter = vectors.iterator();
        while(iter.hasNext())
            if(iter.next().alpha <= EPSILON)
                iter.remove();
    }
    
    /**
     * Return the number of support vectors.
     * @return  the number of support vectors
     */
    public int size() {
        return vectors.size();
    }
    
    /**
     * Calculate the output of the SVM.
     * 
     * @param x  the input vector (x)
     * @return   the output (u)
     */
    public double output(DataVector x) {
        // $u = \sum_j \alpha_j y_j K(x_j, x) - b$
        double u = -b;
        for(SupportVector v : vectors) {
            if(v.alpha <= EPSILON)
                // ignore non-support vectors that have not yet been pruned
                // this is only necessary to improve performance during
                // training, and doesn't do anything once prune() has been
                // called
                continue;
            u += v.alpha * v.y * kernel.getValue(v.x, x);
        }
        return u;
    }
    
    /**
     * Train a Gaussian SVM to learn the noisy XOR function.
     */
    public static void main(String[] args) {
        // input-output pairs for the XOR function
        double[][] xs = {{-1,-1},{+1,-1},{-1,+1},{+1,+1}};
        int[]      ys = {    -1,     +1,     +1,     -1 };
        
        // create a new Gaussian SVM with variance=1,C=100
        SVM svm = new SVM(new GaussianKernel(1), 100);
        Random random = new Random();
        
        // add 50 noisy examples of each input-output pair
        for(int i = 0; i < 4; i++) {
            double[] x = xs[i];
            int y = ys[i];
            for(int j = 0; j < 50; j++) {
                // add some random noise to the input
                DataVector v = new RealVector(
                        x[0] + random.nextGaussian()/2,
                        x[1] + random.nextGaussian()/2);
                // add this example to the svm
                svm.add(v, y); // input vector v belonging to class y
            }
        }
        
        // train the SVM with the SMO technique
        SMO.train(svm);
        
        // print the outputs of the SVM
        for(double[] x : xs) {
            // convert array to vector
            DataVector v = new RealVector(x);
            // calculate output of SVM
            double u = svm.output(v);
            System.out.println(v + " : " + u);
        }
    }
}
