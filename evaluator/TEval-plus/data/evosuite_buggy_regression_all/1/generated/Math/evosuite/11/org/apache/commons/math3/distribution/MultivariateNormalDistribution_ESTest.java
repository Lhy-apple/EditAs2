/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:00:58 GMT 2023
 */

package org.apache.commons.math3.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MultivariateNormalDistribution_ESTest extends MultivariateNormalDistribution_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      double[][] doubleArray1 = new double[3][0];
      MultivariateNormalDistribution multivariateNormalDistribution0 = null;
      try {
        multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 0 != 3
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = (-438.19603501);
      doubleArray0[1] = 2551.186382593726;
      doubleArray0[2] = 1672.737874114;
      double[][] doubleArray1 = new double[3][0];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      doubleArray1[2] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double double0 = multivariateNormalDistribution0.density(doubleArray0);
      assertEquals(9.56638233871764E9, double0, 0.01);
      assertArrayEquals(new double[] {(-438.19603501), 2551.186382593726, 1672.737874114}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = (-438.19603501);
      doubleArray0[1] = 2551.186382593726;
      doubleArray0[2] = 1672.737874114;
      double[][] doubleArray1 = new double[3][0];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      doubleArray1[2] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      Array2DRowRealMatrix array2DRowRealMatrix0 = (Array2DRowRealMatrix)multivariateNormalDistribution0.getCovariances();
      assertFalse(array2DRowRealMatrix0.isTransposable());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      double[] doubleArray0 = new double[13];
      double[][] doubleArray1 = new double[4][7];
      MultivariateNormalDistribution multivariateNormalDistribution0 = null;
      try {
        multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 4 != 13
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = (-438.19603501);
      doubleArray0[1] = 2551.186382593726;
      double[][] doubleArray1 = new double[3][8];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      doubleArray1[2] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = null;
      try {
        multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -0 is smaller than, or equal to, the minimum (0): not positive definite matrix: value -0 at index 1
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = (-438.19603501);
      doubleArray0[1] = 2551.186382593726;
      doubleArray0[2] = 1672.737874114;
      double[][] doubleArray1 = new double[3][0];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      doubleArray1[2] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double[] doubleArray2 = new double[7];
      try { 
        multivariateNormalDistribution0.density(doubleArray2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 7 != 3
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = (-438.19603501);
      doubleArray0[1] = 2551.186382593726;
      doubleArray0[2] = 1672.737874114;
      double[][] doubleArray1 = new double[3][0];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      doubleArray1[2] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double[] doubleArray2 = multivariateNormalDistribution0.getStandardDeviations();
      assertArrayEquals(new double[] {Double.NaN, 50.50927026392013, 40.89911825594777}, doubleArray2, 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[0] = (-438.19603501);
      doubleArray0[1] = 2551.186382593726;
      doubleArray0[2] = 1672.737874114;
      double[][] doubleArray1 = new double[3][0];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      doubleArray1[2] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double[] doubleArray2 = multivariateNormalDistribution0.sample();
      assertArrayEquals(new double[] {(-472.55678880522424), 2516.82561373301, 1638.377108263919}, doubleArray2, 0.01);
  }
}