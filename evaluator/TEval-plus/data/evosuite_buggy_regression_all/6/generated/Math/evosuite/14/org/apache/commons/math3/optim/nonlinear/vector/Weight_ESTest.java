/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:47:56 GMT 2023
 */

package org.apache.commons.math3.optim.nonlinear.vector;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.optim.nonlinear.vector.Weight;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Weight_ESTest extends Weight_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      Weight weight0 = new Weight(doubleArray0);
      RealMatrix realMatrix0 = weight0.getWeight();
      assertEquals(3, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Array2DRowRealMatrix array2DRowRealMatrix0 = new Array2DRowRealMatrix();
      Weight weight0 = null;
      try {
        weight0 = new Weight(array2DRowRealMatrix0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // matrix must have at least one row
         //
         verifyException("org.apache.commons.math3.linear.Array2DRowRealMatrix", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BlockRealMatrix blockRealMatrix0 = new BlockRealMatrix(1140, 8);
      Weight weight0 = null;
      try {
        weight0 = new Weight(blockRealMatrix0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // non square (8x1,140) matrix
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.vector.Weight", e);
      }
  }
}
