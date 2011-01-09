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

package cc.vidr.jclassification.svm.kernel;

import java.io.Serializable;

import cc.vidr.jclassification.svm.vector.DataVector;

/**
 * An interface for kernel functions, since Java doesn't support
 * first-class functions.
 * 
 * @author  David A Roberts
 */
public interface Kernel extends Serializable {
    /**
     * Get the value of the kernel function for the two vectors.
     * @param x1  the first vector
     * @param x2  the second vector
     * @return    the value
     */
    double getValue(DataVector x1, DataVector x2);
}
