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

package cc.vidr.jclassification.util;

/**
 * Various mathematical utility functions.
 * 
 * @author  David A Roberts
 */
public class MathUtil {
    /**
     * Is x == y to within the given tolerance?
     */
    public static boolean equals(double x, double y, double epsilon) {
        return Math.abs(x - y) < epsilon;
    }
    
    /**
     * Is x <= y to within the given tolerance?
     */
    public static boolean leq(double x, double y, double epsilon) {
        return x <= y + epsilon;
    }
    
    /**
     * Is x >= y to within the given tolerance?
     */
    public static boolean geq(double x, double y, double epsilon) {
        return x >= y - epsilon;
    }
    
    /**
     * Clamp the given value to a given range.
     * If it falls outside the range, it is set to the closest bound.
     * 
     * @param x     the value to clamp
     * @param low   the lower bound
     * @param high  the upper bound
     * @return      the clamped value (low <= x <= high)
     */
    public static double clamp(double x, double low, double high) {
        return Math.min(Math.max(x, low), high);
    }
}
