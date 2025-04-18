/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:12:51 GMT 2023
 */

package org.apache.commons.math.optimization;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.UnivariateRealFunction;
import org.apache.commons.math.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math.optimization.GoalType;
import org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer;
import org.apache.commons.math.optimization.UnivariateRealOptimizer;
import org.apache.commons.math.random.MersenneTwister;
import org.apache.commons.math.random.RandomAdaptor;
import org.apache.commons.math.random.RandomGenerator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MultiStartUnivariateRealOptimizer_ESTest extends MultiStartUnivariateRealOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      int[] intArray0 = new int[5];
      MersenneTwister mersenneTwister0 = new MersenneTwister(intArray0);
      RandomAdaptor randomAdaptor0 = new RandomAdaptor(mersenneTwister0);
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, Integer.MAX_VALUE, randomAdaptor0);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.setRelativeAccuracy((-1879.58));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (-1), mersenneTwister0);
      int int0 = multiStartUnivariateRealOptimizer0.getMaximalIterationCount();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(0.0).when(univariateRealOptimizer0).getRelativeAccuracy();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1892, (RandomGenerator) null);
      multiStartUnivariateRealOptimizer0.getRelativeAccuracy();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1892, (RandomGenerator) null);
      multiStartUnivariateRealOptimizer0.setAbsoluteAccuracy(1892);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(0.0).when(univariateRealOptimizer0).getAbsoluteAccuracy();
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 113, mersenneTwister0);
      multiStartUnivariateRealOptimizer0.getAbsoluteAccuracy();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (-1), mersenneTwister0);
      int int0 = multiStartUnivariateRealOptimizer0.getMaxEvaluations();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, int0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn((-354.26)).when(univariateRealOptimizer0).getResult();
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1944, mersenneTwister0);
      multiStartUnivariateRealOptimizer0.getResult();
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MersenneTwister mersenneTwister0 = new MersenneTwister((int[]) null);
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1, mersenneTwister0);
      multiStartUnivariateRealOptimizer0.resetRelativeAccuracy();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 113, mersenneTwister0);
      multiStartUnivariateRealOptimizer0.resetMaximalIterationCount();
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1944, mersenneTwister0);
      multiStartUnivariateRealOptimizer0.resetAbsoluteAccuracy();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(28, 28, 1790, 1310, 1310).when(univariateRealOptimizer0).getIterationCount();
      doReturn(1790, 28, 1790, 1310, 1790).when(univariateRealOptimizer0).getEvaluations();
      doReturn((-2239.224500023194), (-2239.224500023194), (-2239.224500023194), Double.POSITIVE_INFINITY, 2342.8494838735).when(univariateRealOptimizer0).getFunctionValue();
      doReturn((double)1310, 2342.8494838735, (-2239.224500023194), (-1890.0), (double)1790).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 113, mersenneTwister0);
      multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) null, goalType0, 2407.256987, 4064.0, 385.5913938171);
      multiStartUnivariateRealOptimizer0.getOptima();
      assertEquals(2342.8494838735, multiStartUnivariateRealOptimizer0.getFunctionValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1944, mersenneTwister0);
      try { 
        multiStartUnivariateRealOptimizer0.getOptima();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no optimum computed yet
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MersenneTwister mersenneTwister0 = new MersenneTwister((int[]) null);
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(univariateRealOptimizer0).getIterationCount();
      doReturn(0).when(univariateRealOptimizer0).getEvaluations();
      doReturn(0.0).when(univariateRealOptimizer0).getFunctionValue();
      doReturn(0.0).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1, mersenneTwister0);
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      
      double[] doubleArray0 = new double[4];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      GoalType goalType0 = GoalType.MINIMIZE;
      multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) polynomialFunction0, goalType0, 2315.0, (-1.0), 2315.0);
      double[] doubleArray1 = multiStartUnivariateRealOptimizer0.getOptimaValues();
      assertEquals(1, doubleArray1.length);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getEvaluations());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (byte)10, (RandomGenerator) null);
      try { 
        multiStartUnivariateRealOptimizer0.getOptimaValues();
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // no optimum computed yet
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(28, (-2564), (-1004), 5, (-2564)).when(univariateRealOptimizer0).getIterationCount();
      doReturn(28, 30, (-2564), (-1004), 30).when(univariateRealOptimizer0).getEvaluations();
      doReturn((double)28, 3492.5, (double)28, Double.NaN, (double)28).when(univariateRealOptimizer0).getFunctionValue();
      doReturn(Double.NaN, 892.10653039, (-1250.955165452), (double)30, (-1250.955165452)).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 113, mersenneTwister0);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) polynomialFunction0, goalType0, (double) 113, (double) 1, Double.NaN);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 113
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      double[] doubleArray0 = new double[4];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      MersenneTwister mersenneTwister0 = new MersenneTwister((-2713));
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(0, 0, 0, 0, 0).when(univariateRealOptimizer0).getIterationCount();
      doReturn(0, 0, 0, 0, 0).when(univariateRealOptimizer0).getEvaluations();
      doReturn(0.0, 0.0, 0.0, 0.0, 0.0).when(univariateRealOptimizer0).getFunctionValue();
      doReturn(0.0, 0.0, 0.0, 0.0, 0.0).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 1, (RandomGenerator) null);
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer1 = new MultiStartUnivariateRealOptimizer(multiStartUnivariateRealOptimizer0, 28, mersenneTwister0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      multiStartUnivariateRealOptimizer1.optimize((UnivariateRealFunction) polynomialFunction0, goalType0, 1319.29, 2688.566369762884, 1319.29);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
  }
}
