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

package cc.vidr.jclassification.svm.vector;

import java.util.Arrays;

/**
 * A vector of real numbers (approximated by doubles).
 * 
 * @author  David A Roberts
 */
public class RealVector implements DataVector {
    private static final long serialVersionUID = -8736795465347538756L;
    private final double[] vector;
    
    public RealVector(double... vector) {
        this.vector = vector;
    }
    
    public double dotProduct(DataVector x) {
        return dotProduct((RealVector) x);
    }
    
    public double dotProduct(RealVector x) {
        double prod = 0;
        for(int i = 0; i < vector.length; i++)
            prod += vector[i] * x.vector[i];
        return prod;
    }
    
    public double sqDist(DataVector x) {
        return sqDist((RealVector) x);
    }
    
    public double sqDist(RealVector x) {
        double r2 = 0;
        for(int i = 0; i < vector.length; i++) {
            double d = vector[i] - x.vector[i];
            r2 += d*d;
        }
        return r2;
    }
    
    public String toString() {
        return Arrays.toString(vector);
    }
}
