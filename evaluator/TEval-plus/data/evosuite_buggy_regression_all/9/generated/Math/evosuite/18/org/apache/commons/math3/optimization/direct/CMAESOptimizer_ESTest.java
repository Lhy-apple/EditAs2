/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:37:26 GMT 2023
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
import org.apache.commons.math3.optimization.direct.CMAESOptimizer;
import org.apache.commons.math3.random.UnitSphereRandomVectorGenerator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CMAESOptimizer_ESTest extends CMAESOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 990, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(990, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(0, doubleArray0);
      List<Double> list0 = cMAESOptimizer0.getStatisticsSigmaHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsMeanHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(616);
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsDHistory();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(0);
      List<Double> list0 = cMAESOptimizer0.getStatisticsFitnessHistory();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 990, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(32, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 990, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer((-1569), doubleArray1, (-1569), 32, true, (-1), 30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false);
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(32, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 12, (-1.2711589287782304E-7), false, 0, (-2753), cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(188, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 30000, 0, true, 30000, 30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(25, doubleArray0[0], 25, 30000, true, 25, 25, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      // Undeclared exception!
      cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray0[0]);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(25, doubleArray0[0], 25, 30000, true, 25, 25, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      // Undeclared exception!
      cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray0[0]);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize((-1168), (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, (double[]) null, doubleArray1);
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
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(2266, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, (double[]) null);
        fail("Expecting exception: MathUnsupportedOperationException");
      
      } catch(MathUnsupportedOperationException e) {
         //
         // unsupported operation
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(25, doubleArray0[0], 25, 30000, true, 25, 25, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 8 != 1
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[][] doubleArray0 = new double[1][8];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      double[] doubleArray1 = new double[1];
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      double[] doubleArray2 = new double[7];
      doubleArray2[0] = (-1796.68787732158);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(0, doubleArray2);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(408074248, microsphereInterpolatingFunction0, goalType0, doubleArray2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1,796.688 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray1[0] = (double) 30000;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 990, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer((-1569), doubleArray1, (-1569), 32, true, (-1), 30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false);
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(67488986, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 30,000 out of [0, 0] range
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 30000, 0, false, (-744), 32, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(925, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][8];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(30000, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
      assertNotNull(pointValuePair0);
  }
}
