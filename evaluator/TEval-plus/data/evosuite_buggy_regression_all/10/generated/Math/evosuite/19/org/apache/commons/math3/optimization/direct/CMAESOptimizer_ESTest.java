/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:20:41 GMT 2023
 */

package org.apache.commons.math3.optimization.direct;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.analysis.interpolation.MicrosphereInterpolatingFunction;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.optimization.GoalType;
import org.apache.commons.math3.optimization.PointValuePair;
import org.apache.commons.math3.optimization.SimpleValueChecker;
import org.apache.commons.math3.optimization.direct.CMAESOptimizer;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.UnitSphereRandomVectorGenerator;
import org.apache.commons.math3.random.Well19937a;
import org.apache.commons.math3.random.Well44497b;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CMAESOptimizer_ESTest extends CMAESOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      doubleArray1[0] = doubleArray0;
      double[] doubleArray2 = new double[1];
      doubleArray2[0] = (double) 30000;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray2, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(1969, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray0, doubleArray0, doubleArray2);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<Double> list0 = cMAESOptimizer0.getStatisticsSigmaHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(1025, (double[]) null);
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsDHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsMeanHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<Double> list0 = cMAESOptimizer0.getStatisticsFitnessHistory();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(331, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      double[][] doubleArray1 = new double[1][7];
      doubleArray1[0] = doubleArray0;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray0, 0, 0, true, 331, 30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(32, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray0, doubleArray0, doubleArray0);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(331, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      double[][] doubleArray1 = new double[1][7];
      doubleArray1[0] = doubleArray0;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(60, doubleArray0, 331, (-1582.1780941149), true, 1421, (-345), cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(439, microsphereInterpolatingFunction0, goalType0, doubleArray0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      doubleArray1[0] = doubleArray0;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      Well44497b well44497b0 = new Well44497b();
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(30000, 2.0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(110, doubleArray0, 110, 0.0, true, 110, 3278, well44497b0, true, simpleValueChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray0);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      doubleArray1[0] = doubleArray0;
      double[] doubleArray2 = new double[1];
      doubleArray2[0] = (double) 30000;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray2, 0, 0, unitSphereRandomVectorGenerator0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker((-2.4921733203932487E-10), 0.02745391119371794);
      int[] intArray0 = new int[1];
      Well19937a well19937a0 = new Well19937a(intArray0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray2, 52, 5.688906371296133E-247, false, 0, 2725, well19937a0, false, simpleValueChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(1329, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray0, doubleArray0, doubleArray2);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      double[][] doubleArray1 = new double[1][7];
      doubleArray1[0] = doubleArray0;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(0, 30000);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray0, 30000, 214.60206021759953, true, 0, 0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true, simpleValueChecker0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray0);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(1, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray0, (double[]) null, doubleArray0);
        fail("Expecting exception: MathUnsupportedOperationException");
      
      } catch(MathUnsupportedOperationException e) {
         //
         // unsupported operation
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      doubleArray1[0] = doubleArray0;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray1[0], 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(0, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1[0], doubleArray1[0], (double[]) null);
        fail("Expecting exception: MathUnsupportedOperationException");
      
      } catch(MathUnsupportedOperationException e) {
         //
         // unsupported operation
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      double[][] doubleArray1 = new double[1][7];
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(0, 30000);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1[0], 30000, 0.0, true, 0, 0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true, simpleValueChecker0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(0, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray0, doubleArray0, doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 7 != 1
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(1.0, 1.0);
      double[] doubleArray2 = new double[7];
      doubleArray2[3] = (-509.566);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(14, doubleArray2, 0, (-516.53573143623), true, 0, 0, (RandomGenerator) null, false, simpleValueChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -509.566 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      double[] doubleArray2 = new double[1];
      doubleArray2[0] = (double) 1969;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray2, 0, 0, unitSphereRandomVectorGenerator0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(0, 0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer((-5), doubleArray2, 2, 2806.370769365, true, 1969, 2, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false, simpleValueChecker0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize((-1320869381), (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray0, doubleArray0, doubleArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 1,969 out of [0, 0] range
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      doubleArray1[0] = doubleArray0;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray0, 0, 0, unitSphereRandomVectorGenerator0);
      Well44497b well44497b0 = new Well44497b();
      well44497b0.setSeed((long) 0);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(30000, 2.0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(110, doubleArray0, 110, 0.0, true, 110, 3278, well44497b0, true, simpleValueChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray0);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[] doubleArray0 = new double[1];
      double[][] doubleArray1 = new double[1][5];
      doubleArray1[0] = doubleArray0;
      double[] doubleArray2 = new double[1];
      doubleArray2[0] = (double) 30000;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray1, doubleArray2, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(1969, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray2, doubleArray0, doubleArray2);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][5];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      cMAESOptimizer0.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray0[0]);
  }
}