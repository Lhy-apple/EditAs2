/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:01:07 GMT 2023
 */

package org.apache.commons.math3.optim.nonlinear.scalar.noderiv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.SimplePointChecker;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well1024a;
import org.apache.commons.math3.random.Well19937a;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.random.Well44497b;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CMAESOptimizer_ESTest extends CMAESOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker((-2292), (-2292));
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer((-2292), (-2292), false, (-2292), (-2292), (RandomGenerator) null, false, simpleValueChecker0);
      List<Double> list0 = cMAESOptimizer0.getStatisticsSigmaHistory();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Well19937c well19937c0 = new Well19937c(0);
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>(101, 0, 101);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(101, 101, true, 101, 101, well19937c0, true, simplePointChecker0);
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsDHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      int[] intArray0 = new int[0];
      Well19937a well19937a0 = new Well19937a(intArray0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker((-3210), (-3210));
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer((-3210), (-3210), true, (-3210), (-3210), well19937a0, true, simpleValueChecker0);
      List<Double> list0 = cMAESOptimizer0.getStatisticsFitnessHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(266, 266);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(266, 266, false, 266, 266, (RandomGenerator) null, false, simpleValueChecker0);
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsMeanHistory();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      doubleArray0[1] = (double) (-4987);
      CMAESOptimizer.Sigma cMAESOptimizer_Sigma0 = null;
      try {
        cMAESOptimizer_Sigma0 = new CMAESOptimizer.Sigma(doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -4,987 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer$Sigma", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      CMAESOptimizer.PopulationSize cMAESOptimizer_PopulationSize0 = null;
      try {
        cMAESOptimizer_PopulationSize0 = new CMAESOptimizer.PopulationSize((-1037));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1,037 is smaller than, or equal to, the minimum (0)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer$PopulationSize", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(266, 266);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer((-1023), 266, true, 937, 266, (RandomGenerator) null, true, simpleValueChecker0);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[4];
      CMAESOptimizer.PopulationSize cMAESOptimizer_PopulationSize0 = new CMAESOptimizer.PopulationSize(266);
      optimizationDataArray0[3] = (OptimizationData) cMAESOptimizer_PopulationSize0;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      CMAESOptimizer.Sigma cMAESOptimizer_Sigma0 = new CMAESOptimizer.Sigma(doubleArray0);
      Well1024a well1024a0 = new Well1024a();
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(2884, 0.0);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(6, 10, true, (-167), (-1567), well1024a0, true, simpleValueChecker0);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[5];
      optimizationDataArray0[1] = (OptimizationData) cMAESOptimizer_Sigma0;
      double[] doubleArray1 = new double[1];
      InitialGuess initialGuess0 = new InitialGuess(doubleArray1);
      optimizationDataArray0[4] = (OptimizationData) initialGuess0;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.parseOptimizationData(optimizationDataArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 5 != 1
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      CMAESOptimizer.Sigma cMAESOptimizer_Sigma0 = new CMAESOptimizer.Sigma(doubleArray0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(2884, (-2.5676078228301587E-8));
      Well44497b well44497b0 = new Well44497b(9223372036854775807L);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(6, 752.0, true, 2884, 6, well44497b0, true, simpleValueChecker0);
      InitialGuess initialGuess0 = new InitialGuess(doubleArray0);
      SimpleBounds simpleBounds0 = new SimpleBounds(doubleArray0, doubleArray0);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[4];
      optimizationDataArray0[0] = (OptimizationData) initialGuess0;
      optimizationDataArray0[2] = (OptimizationData) simpleBounds0;
      optimizationDataArray0[3] = (OptimizationData) cMAESOptimizer_Sigma0;
      cMAESOptimizer0.parseOptimizationData(optimizationDataArray0);
      assertEquals(4, optimizationDataArray0.length);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      doubleArray0[0] = (double) 2884;
      CMAESOptimizer.Sigma cMAESOptimizer_Sigma0 = new CMAESOptimizer.Sigma(doubleArray0);
      Well1024a well1024a0 = new Well1024a(6);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(2884, 2884.0);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(6, 2884.0, true, 2884, 33, well1024a0, true, simpleValueChecker0);
      InitialGuess initialGuess0 = new InitialGuess(doubleArray0);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[4];
      SimpleBounds simpleBounds0 = new SimpleBounds(doubleArray0, doubleArray0);
      optimizationDataArray0[0] = (OptimizationData) simpleBounds0;
      optimizationDataArray0[1] = (OptimizationData) initialGuess0;
      optimizationDataArray0[3] = (OptimizationData) cMAESOptimizer_Sigma0;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.parseOptimizationData(optimizationDataArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 2,884 out of [0, 0] range
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer", e);
      }
  }
}