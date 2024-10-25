/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:49:02 GMT 2023
 */

package org.apache.commons.math.linear;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.linear.CholeskyDecompositionImpl;
import org.apache.commons.math.linear.DecompositionSolver;
import org.apache.commons.math.linear.DenseRealMatrix;
import org.apache.commons.math.linear.OpenMapRealVector;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealMatrixImpl;
import org.apache.commons.math.linear.RealVector;
import org.apache.commons.math.linear.RealVectorImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CholeskyDecompositionImpl_ESTest extends CholeskyDecompositionImpl_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      RealMatrixImpl realMatrixImpl0 = new RealMatrixImpl();
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(realMatrixImpl0);
      DecompositionSolver decompositionSolver0 = choleskyDecompositionImpl0.getSolver();
      assertTrue(decompositionSolver0.isNonSingular());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      double[][] doubleArray0 = new double[1][3];
      DenseRealMatrix denseRealMatrix0 = new DenseRealMatrix(doubleArray0);
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = null;
      try {
        choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(denseRealMatrix0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
         //
         // a 1x3 matrix was provided instead of a square matrix
         //
         verifyException("org.apache.commons.math.linear.CholeskyDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      RealMatrixImpl realMatrixImpl0 = new RealMatrixImpl(2656, 2656);
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = null;
      try {
        choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(realMatrixImpl0);
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // not positive definite matrix
         //
         verifyException("org.apache.commons.math.linear.CholeskyDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      RealMatrixImpl realMatrixImpl0 = new RealMatrixImpl(31, 31);
      RealMatrix realMatrix0 = realMatrixImpl0.scalarAdd(31);
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(realMatrix0);
      assertEquals(Double.NaN, choleskyDecompositionImpl0.getDeterminant(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1140.83219043811;
      doubleArray0[1] = 1140.83219043811;
      RealVectorImpl realVectorImpl0 = new RealVectorImpl(doubleArray0);
      RealMatrix realMatrix0 = realVectorImpl0.outerProduct(realVectorImpl0);
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = null;
      try {
        choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(realMatrix0, (-273.13759487), (-1.0));
        fail("Expecting exception: Exception");
      
      } catch(Throwable e) {
         //
         // not symmetric matrix
         //
         verifyException("org.apache.commons.math.linear.CholeskyDecompositionImpl", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      RealVectorImpl realVectorImpl0 = new RealVectorImpl(doubleArray0, false);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(realVectorImpl0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct((RealVector) realVectorImpl0);
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(realMatrix0, 1.0E-12, (-13.50736469232692));
      choleskyDecompositionImpl0.getL();
      RealMatrix realMatrix1 = choleskyDecompositionImpl0.getL();
      assertNotNull(realMatrix1);
      assertEquals(Double.NaN, realMatrix1.getFrobeniusNorm(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      RealVectorImpl realVectorImpl0 = new RealVectorImpl(doubleArray0, false);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(realVectorImpl0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct((RealVector) realVectorImpl0);
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(realMatrix0, 1.0E-12, (-13.50736469232692));
      choleskyDecompositionImpl0.getLT();
      RealMatrix realMatrix1 = choleskyDecompositionImpl0.getL();
      assertEquals(Double.NaN, realMatrix1.getFrobeniusNorm(), 0.01);
      assertNotNull(realMatrix1);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      RealVectorImpl realVectorImpl0 = new RealVectorImpl(doubleArray0, false);
      OpenMapRealVector openMapRealVector0 = new OpenMapRealVector(realVectorImpl0);
      RealMatrix realMatrix0 = openMapRealVector0.outerProduct((RealVector) realVectorImpl0);
      CholeskyDecompositionImpl choleskyDecompositionImpl0 = new CholeskyDecompositionImpl(realMatrix0, 1.0E-12, (-13.50736469232692));
      double double0 = choleskyDecompositionImpl0.getDeterminant();
      assertEquals(Double.NaN, double0, 0.01);
  }
}
