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

/**
 * A vector optimised for binary elements. Much more efficient than
 * RealVector.
 * 
 * @author  David A Roberts
 */
public class BitVector implements DataVector {
    private static final long serialVersionUID = 8184851900633995703L;
    private long[] vector;
    
    public BitVector(boolean... vector) {
        this(pack(vector));
    }
    
    public BitVector(long... vector) {
        this.vector = vector;
    }
    
    /**
     * Pack each block of 64 bits into a long.
     * 
     * @param bits  the array of bits
     * @return      an array of longs
     */
    private static long[] pack(boolean[] bits) {
        long[] vector = new long[bits.length/Long.SIZE];
        for(int i = 0; i < vector.length; i++) {
            for(int j = 0; j < Long.SIZE; j++) {
                boolean bit = bits[i*Long.SIZE + j];
                if(bit) vector[i] |= 1<<j;
            }
        }
        return vector;
    }
    
    public double dotProduct(DataVector x) {
        return dotProduct((BitVector) x);
    }
    
    public int dotProduct(BitVector x) {
        int prod = 0;
        for(int i = 0; i < vector.length; i++)
            // Hamming weight of ANDed vectors
            prod += Long.bitCount(vector[i] & x.vector[i]);
        return prod;
    }
    
    public double sqDist(DataVector x) {
        return hammingDist((BitVector) x);
    }
    
    /**
     * Returns the Hamming distance of the two vectors.
     */
    public int hammingDist(BitVector x) {
        int dist = 0;
        for(int i = 0; i < vector.length; i++)
            // Hamming weight of XORed vectors
            dist += Long.bitCount(vector[i] ^ x.vector[i]);
        return dist;
    }
}
