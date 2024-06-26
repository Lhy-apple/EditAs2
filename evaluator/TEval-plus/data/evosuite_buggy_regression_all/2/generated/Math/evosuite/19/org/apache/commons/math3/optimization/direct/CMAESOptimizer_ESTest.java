/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:26:17 GMT 2023
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
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, (UnitSphereRandomVectorGenerator) null);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(433, microsphereInterpolatingFunction0, goalType0, doubleArray1);
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
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer((-806));
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsMeanHistory();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer((-806));
      List<RealMatrix> list0 = cMAESOptimizer0.getStatisticsDHistory();
      assertTrue(list0.isEmpty());
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
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, (UnitSphereRandomVectorGenerator) null);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(30000, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, (UnitSphereRandomVectorGenerator) null);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(4, doubleArray1, 0, 0.12113528959154087, true, 30000, 0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(4, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, (UnitSphereRandomVectorGenerator) null);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 712, 86.80966689574, true, 30000, (-59), cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer1.optimize(3672, microsphereInterpolatingFunction0, goalType0, doubleArray1);
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
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 479, 0, (UnitSphereRandomVectorGenerator) null);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 30000, 479, true, 479, 30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(30000, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, (UnitSphereRandomVectorGenerator) null);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(0, doubleArray1, 30000, 30000, true, 30000, 30000, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, true);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(8, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, (UnitSphereRandomVectorGenerator) null);
      GoalType goalType0 = GoalType.MAXIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer0.optimize(1707, microsphereInterpolatingFunction0, goalType0, doubleArray1);
      assertNotNull(pointValuePair0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer();
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, (UnitSphereRandomVectorGenerator) null);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(2184, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, (double[]) null, doubleArray1);
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
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, (UnitSphereRandomVectorGenerator) null);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(0, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, (double[]) null);
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
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 30000, 0, (UnitSphereRandomVectorGenerator) null);
      double[] doubleArray2 = new double[0];
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(0, doubleArray1);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(30000, microsphereInterpolatingFunction0, goalType0, doubleArray2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 1 != 0
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, (UnitSphereRandomVectorGenerator) null);
      double[] doubleArray2 = new double[1];
      doubleArray2[0] = (double) (-111);
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(0, doubleArray2);
      GoalType goalType0 = GoalType.MAXIMIZE;
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(0, microsphereInterpolatingFunction0, goalType0, doubleArray2);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -111 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.direct.CMAESOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray1[0] = (double) 30000;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 20, 0, (UnitSphereRandomVectorGenerator) null);
      GoalType goalType0 = GoalType.MINIMIZE;
      CMAESOptimizer cMAESOptimizer0 = new CMAESOptimizer(464, doubleArray1);
      // Undeclared exception!
      try { 
        cMAESOptimizer0.optimize(20, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
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
      double[][] doubleArray0 = new double[1][2];
      double[] doubleArray1 = new double[1];
      doubleArray0[0] = doubleArray1;
      MicrosphereInterpolatingFunction microsphereInterpolatingFunction0 = new MicrosphereInterpolatingFunction(doubleArray0, doubleArray1, 0, 0, (UnitSphereRandomVectorGenerator) null);
      CMAESOptimizer cMAESOptimizer1 = new CMAESOptimizer(27, doubleArray1, 2378, 0.0, false, (-1809), 0, cMAESOptimizer0.DEFAULT_RANDOMGENERATOR, false);
      GoalType goalType0 = GoalType.MINIMIZE;
      PointValuePair pointValuePair0 = cMAESOptimizer1.optimize(2000, (MultivariateFunction) microsphereInterpolatingFunction0, goalType0, doubleArray1, doubleArray1, doubleArray1);
      assertNotNull(pointValuePair0);
  }
}
