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

import cc.vidr.jclassification.svm.vector.DataVector;

/**
 * A data class to store an input vector, its target class, and its
 * corresponding Lagrange multiplier.
 * 
 * @author  David A Roberts
 */
public class SupportVector implements Serializable {
    private static final long serialVersionUID = 2454189485098040287L;
    /** The input vector */
    final DataVector x;
    /** The target class: either +1 or -1 */
    final byte y;
    /** The Lagrange multiplier for this example */
    double alpha = 0;
    /** Is the Lagrange multiplier bound? (Only used by SMO) */
    transient boolean bound = true;
    
    public SupportVector(DataVector x, int y) {
        if(Math.abs(y) != 1)
            throw new IllegalArgumentException("y must be either +1 or -1");
        this.x = x;
        this.y = (byte) y;
    }
}