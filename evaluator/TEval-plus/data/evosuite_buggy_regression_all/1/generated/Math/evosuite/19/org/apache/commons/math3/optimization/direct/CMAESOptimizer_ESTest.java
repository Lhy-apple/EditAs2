/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:03:17 GMT 2023
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
import org.apache.commons.math3.optimization.SimplePointChecker;
import org.apache.commons.math3.optimization.SimpleValueChecker;
import org.apache.commons.math3.optimization.direct.CMAESOptimizer;
import org.apache.commons.math3.random.UnitSphereRandomVectorGenerator;
import org.apache.commons.math3.random.Well1024a;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CMAESOptimizer_ESTest extends CMAESOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<Double> list0 = cMAESOptimizer0.getStatisticsSigmaHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(70, doubleArray0);
      assertEquals(0.0, CMAESOptimizer.DEFAULT_STOPFITNESS, 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsMeanHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsDHistory();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      List<Double> list0 = cMAESOptimizer0.getStatisticsFitnessHistory();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer((-404), doubleArray1, (-404), (-404), true, 0, 6043, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(5, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>((-1692.9711), (-182519384));
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer((-182519384), doubleArray1, 30000, (-1692.9711), true, (-182519384), (-182519384), cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false, simplePointChecker0);
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(201, microsphereInterpolatingFunction0, goalType0, doubleArray1);
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
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>(0, 0);
      Well1024a well1024a0 = new Well1024a(30000);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 30000, 30000, true, 0, 0, well1024a0, true, simplePointChecker0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(1);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 355, 1, unitSphereRandomVectorGenerator0);
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(355, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator1 = new UnitSphereRandomVectorGenerator(1);
      unitSphereRandomVectorGenerator1.nextVector();
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 355, 1, unitSphereRandomVectorGenerator1);
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(355, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray1[0] = (double) 30000;
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>(515.50163, 0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer((-182519384), doubleArray1, 30000, 0.0, true, (-182519384), 0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true, simplePointChecker0);
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(288, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, (double[]) null, doubleArray1);
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
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, (-557085494), 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize((-557085494), (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, (double[]) null);
        fail("Expecting exception: MathUnsupportedOperationException");
      
      } catch(MathUnsupportedOperationException e) {
         //
         // unsupported operation
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      Well1024a well1024a0 = new Well1024a();
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(30000, 0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 0, 0, false, 0, 2979, well1024a0, true, simpleValueChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray0[0]);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 1 != 9
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>(515.50163, 0);
      Well1024a well1024a0 = new Well1024a(1);
      double[] doubleArray2 = new double[2];
      doubleArray2[0] = (double) (-182519384);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray2, (-46), 30000, true, (-62), 1283, well1024a0, true, simplePointChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize((-1469727044), microsphereInterpolatingFunction0, goalType0, doubleArray2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -182,519,384 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray1[0] = (double) 30000;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 204, 0, unitSphereRandomVectorGenerator0);
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>(1.597259768214821E167, 30000.0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 18, 1.0, false, 781, 18, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false, simplePointChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(605, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 30,000 out of [0, 0] range
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      Well1024a well1024a0 = new Well1024a();
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(0.66, 7);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 7, 0, false, 7, 2979, well1024a0, true, simpleValueChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      PointValuePair pointValuePair1 = cMAESOptimizer1.doOptimize();
      assertFalse(pointValuePair1.equals((Object)pointValuePair0));
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MINIMIZE;
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>(30000, 0);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer((-182519384), doubleArray1, 30000, 1215.29905282342, false, (-182519384), 0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false, simplePointChecker0);
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      double[][] doubleArray0 = new double[1][9];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, unitSphereRandomVectorGenerator0);
      SimplePointChecker<PointValuePair> simplePointChecker0 = new SimplePointChecker<PointValuePair>(982.9619860199128, 982.9619860199128);
      Well1024a well1024a0 = new Well1024a(30000);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(0, doubleArray1, 30000, 0, true, 0, 30000, well1024a0, true, simplePointChecker0);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(1388, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      UnitSphereRandomVectorGenerator unitSphereRandomVectorGenerator0 = new UnitSphereRandomVectorGenerator(0);
      double[][] doubleArray0 = new double[1][5];
      double[] doubleArray1 = new double[7];
      doubleArray0[0] = doubleArray1;
      double[] doubleArray2 = new double[1];
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray2, 2253, 0, unitSphereRandomVectorGenerator0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      cMAESOptimizer0.optimize(30005, microsphereInterpolatingFunction0, goalType0, doubleArray1);
  }
}