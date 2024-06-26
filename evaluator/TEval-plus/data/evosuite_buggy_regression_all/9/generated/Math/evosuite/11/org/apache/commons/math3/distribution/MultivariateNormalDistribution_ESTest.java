/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:34:18 GMT 2023
 */

package org.apache.commons.math3.distribution;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.RealMatrix;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MultivariateNormalDistribution_ESTest extends MultivariateNormalDistribution_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      double[][] doubleArray1 = new double[2][4];
      MultivariateNormalDistribution multivariateNormalDistribution0 = null;
      try {
        multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 4 != 2
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1.7331079674066365E-9;
      doubleArray0[1] = 1.7331079674066365E-9;
      double[][] doubleArray1 = new double[2][4];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      RealMatrix realMatrix0 = multivariateNormalDistribution0.getCovariances();
      assertEquals(2, realMatrix0.getColumnDimension());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      double[][] doubleArray1 = new double[10][2];
      MultivariateNormalDistribution multivariateNormalDistribution0 = null;
      try {
        multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 10 != 2
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1.7331079674066365E-9;
      doubleArray0[1] = (-1754.6381637514);
      double[][] doubleArray1 = new double[2][4];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = null;
      try {
        multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1,754.638 is smaller than, or equal to, the minimum (0): not positive definite matrix: value -1,754.638 at index 1
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1.7331079674066365E-9;
      doubleArray0[1] = 1.7331079674066365E-9;
      double[][] doubleArray1 = new double[2][4];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double double0 = multivariateNormalDistribution0.density(doubleArray0);
      assertEquals(4.2034622238476495E15, double0, 0.01);
      assertArrayEquals(new double[] {1.7331079674066365E-9, 1.7331079674066365E-9}, doubleArray0, 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1.7331079674066365E-9;
      doubleArray0[1] = 1.7331079674066365E-9;
      double[][] doubleArray1 = new double[2][4];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double[] doubleArray2 = new double[3];
      try { 
        multivariateNormalDistribution0.density(doubleArray2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 3 != 2
         //
         verifyException("org.apache.commons.math3.distribution.MultivariateNormalDistribution", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1.7331079674066365E-9;
      doubleArray0[1] = 1.7331079674066365E-9;
      double[][] doubleArray1 = new double[2][4];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double[] doubleArray2 = multivariateNormalDistribution0.getStandardDeviations();
      assertArrayEquals(new double[] {4.163061334410816E-5, 4.163061334410816E-5}, doubleArray2, 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1.7331079674066365E-9;
      doubleArray0[1] = 1.7331079674066365E-9;
      double[][] doubleArray1 = new double[2][4];
      doubleArray1[0] = doubleArray0;
      doubleArray1[1] = doubleArray0;
      MultivariateNormalDistribution multivariateNormalDistribution0 = new MultivariateNormalDistribution(doubleArray0, doubleArray1);
      double[] doubleArray2 = multivariateNormalDistribution0.sample();
      assertArrayEquals(new double[] {6.931726461108304E-6, 6.9317255977934594E-6}, doubleArray2, 0.01);
  }
}
