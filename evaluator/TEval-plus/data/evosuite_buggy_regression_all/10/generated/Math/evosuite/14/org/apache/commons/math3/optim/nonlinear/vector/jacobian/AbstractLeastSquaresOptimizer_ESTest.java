/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:18:35 GMT 2023
 */

package org.apache.commons.math3.optim.nonlinear.vector.jacobian;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.nonlinear.vector.Weight;
import org.apache.commons.math3.optim.nonlinear.vector.jacobian.LevenbergMarquardtOptimizer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AbstractLeastSquaresOptimizer_ESTest extends AbstractLeastSquaresOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      LevenbergMarquardtOptimizer levenbergMarquardtOptimizer0 = new LevenbergMarquardtOptimizer();
      levenbergMarquardtOptimizer0.setCost((-206.16467579));
      assertEquals(42503.87354359581, levenbergMarquardtOptimizer0.getChiSquare(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      LevenbergMarquardtOptimizer levenbergMarquardtOptimizer0 = new LevenbergMarquardtOptimizer();
      double[] doubleArray0 = new double[2];
      // Undeclared exception!
      try { 
        levenbergMarquardtOptimizer0.computeSigma(doubleArray0, 0.0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.vector.JacobianMultivariateVectorOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      LevenbergMarquardtOptimizer levenbergMarquardtOptimizer0 = new LevenbergMarquardtOptimizer((-1846.8030083185213), (-1846.8030083185213), (-1846.8030083185213));
      // Undeclared exception!
      try { 
        levenbergMarquardtOptimizer0.getRMS();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.vector.MultivariateVectorOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      LevenbergMarquardtOptimizer levenbergMarquardtOptimizer0 = new LevenbergMarquardtOptimizer((-1861.8520336467348), (-1861.8520336467348), (-1861.8520336467348));
      OptimizationData[] optimizationDataArray0 = new OptimizationData[2];
      double[] doubleArray0 = new double[5];
      Weight weight0 = new Weight(doubleArray0);
      optimizationDataArray0[0] = (OptimizationData) weight0;
      // Undeclared exception!
      try { 
        levenbergMarquardtOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: MathUnsupportedOperationException");
      
      } catch(MathUnsupportedOperationException e) {
         //
         // unsupported operation
         //
         verifyException("org.apache.commons.math3.linear.EigenDecomposition", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      LevenbergMarquardtOptimizer levenbergMarquardtOptimizer0 = new LevenbergMarquardtOptimizer();
      // Undeclared exception!
      try { 
        levenbergMarquardtOptimizer0.getWeightSquareRoot();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.vector.jacobian.AbstractLeastSquaresOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      LevenbergMarquardtOptimizer levenbergMarquardtOptimizer0 = new LevenbergMarquardtOptimizer();
      double[] doubleArray0 = new double[0];
      // Undeclared exception!
      try { 
        levenbergMarquardtOptimizer0.computeCost(doubleArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.vector.MultivariateVectorOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      LevenbergMarquardtOptimizer levenbergMarquardtOptimizer0 = new LevenbergMarquardtOptimizer();
      OptimizationData[] optimizationDataArray0 = new OptimizationData[1];
      // Undeclared exception!
      try { 
        levenbergMarquardtOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.vector.MultivariateVectorOptimizer", e);
      }
  }
}
