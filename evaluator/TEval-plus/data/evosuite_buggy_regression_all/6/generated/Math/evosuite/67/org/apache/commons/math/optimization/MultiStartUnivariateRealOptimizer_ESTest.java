/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:54:00 GMT 2023
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
import org.apache.commons.math.random.JDKRandomGenerator;
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
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (-1), mersenneTwister0);
      multiStartUnivariateRealOptimizer0.setRelativeAccuracy(0.0);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (-1), mersenneTwister0);
      int int0 = multiStartUnivariateRealOptimizer0.getMaximalIterationCount();
      assertEquals(Integer.MAX_VALUE, int0);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      JDKRandomGenerator jDKRandomGenerator0 = new JDKRandomGenerator();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, (-2077), jDKRandomGenerator0);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.getRelativeAccuracy();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MersenneTwister mersenneTwister0 = new MersenneTwister(1097L);
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, 1447, mersenneTwister0);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.setAbsoluteAccuracy(2.0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, 2007, (RandomGenerator) null);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.getAbsoluteAccuracy();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, (-3979), mersenneTwister0);
      int int0 = multiStartUnivariateRealOptimizer0.getMaxEvaluations();
      assertEquals(Integer.MAX_VALUE, int0);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MersenneTwister mersenneTwister0 = new MersenneTwister(0L);
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, (-687), mersenneTwister0);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.getResult();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (-1), mersenneTwister0);
      multiStartUnivariateRealOptimizer0.resetRelativeAccuracy();
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister((long) 0);
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 0, mersenneTwister0);
      int int0 = multiStartUnivariateRealOptimizer0.getIterationCount();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, int0);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(0, 0, 0, 0, 0).when(univariateRealOptimizer0).getIterationCount();
      doReturn(0, 0, 0, 0, 0).when(univariateRealOptimizer0).getEvaluations();
      doReturn(0.0, 0.0, 0.0, 0.0, 0.0).when(univariateRealOptimizer0).getFunctionValue();
      doReturn(0.0, 0.0, 0.0, 0.0, 0.0).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      MersenneTwister mersenneTwister0 = new MersenneTwister(36);
      RandomAdaptor randomAdaptor0 = new RandomAdaptor(mersenneTwister0);
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 25, randomAdaptor0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) null, goalType0, (double) 1, (double) 25, 395.497435);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getEvaluations());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      JDKRandomGenerator jDKRandomGenerator0 = new JDKRandomGenerator();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, 34, jDKRandomGenerator0);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.resetMaximalIterationCount();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (-1), mersenneTwister0);
      multiStartUnivariateRealOptimizer0.resetAbsoluteAccuracy();
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      JDKRandomGenerator jDKRandomGenerator0 = new JDKRandomGenerator();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, 1284, jDKRandomGenerator0);
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.getFunctionValue();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, 2007, (RandomGenerator) null);
      int int0 = multiStartUnivariateRealOptimizer0.getEvaluations();
      assertEquals(0, int0);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(2517, 2517, 2517, 2517, (-482)).when(univariateRealOptimizer0).getIterationCount();
      doReturn(2517, 2517, 2517, 1550, 2517).when(univariateRealOptimizer0).getEvaluations();
      doReturn(1832.3112372066, (double)2517, (double)(-482), (double)2517, 726.4595062994).when(univariateRealOptimizer0).getFunctionValue();
      doReturn((double)1550, (double)1550, (double)1550, (double)(-482), (-113.0)).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      JDKRandomGenerator jDKRandomGenerator0 = new JDKRandomGenerator();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 25, jDKRandomGenerator0);
      double[] doubleArray0 = new double[1];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) polynomialFunction0, goalType0, 43.0, (double) 25);
      multiStartUnivariateRealOptimizer0.getOptima();
      assertEquals(726.4595062994, multiStartUnivariateRealOptimizer0.getFunctionValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, (-1), mersenneTwister0);
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
  public void test16()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn((-265), (-1179), 136, 1, 1867).when(univariateRealOptimizer0).getIterationCount();
      doReturn(136, 1, 1867, (-1629), 21).when(univariateRealOptimizer0).getEvaluations();
      doReturn((-2440.0), (double)1867, (double)1, 1497.0, (-2440.0)).when(univariateRealOptimizer0).getFunctionValue();
      doReturn((double)(-1629), (double)21, 88.1452483528838, (double)21, (double)(-265)).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      MersenneTwister mersenneTwister0 = new MersenneTwister(36);
      RandomAdaptor randomAdaptor0 = new RandomAdaptor(mersenneTwister0);
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 25, randomAdaptor0);
      GoalType goalType0 = GoalType.MAXIMIZE;
      multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) null, goalType0, 2866.870837, (-2392.991));
      multiStartUnivariateRealOptimizer0.getOptimaValues();
      assertEquals(816, multiStartUnivariateRealOptimizer0.getEvaluations());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer((UnivariateRealOptimizer) null, 2007, (RandomGenerator) null);
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
  public void test18()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(56, 56, 56, 56, 56).when(univariateRealOptimizer0).getIterationCount();
      doReturn(269, 269, 269, 56, 4506).when(univariateRealOptimizer0).getEvaluations();
      doReturn((double)4506, (double)4506, (double)269, (double)56, (double)4506).when(univariateRealOptimizer0).getFunctionValue();
      doReturn((double)4506, (-1718.13309254219), (double)269, Double.NaN, (double)56).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      JDKRandomGenerator jDKRandomGenerator0 = new JDKRandomGenerator();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 25, jDKRandomGenerator0);
      double[] doubleArray0 = new double[1];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      GoalType goalType0 = GoalType.MINIMIZE;
      // Undeclared exception!
      try { 
        multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) polynomialFunction0, goalType0, (double) 25, 0.0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 25
         //
         verifyException("org.apache.commons.math.optimization.MultiStartUnivariateRealOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(0, 0, 0, 0, 0).when(univariateRealOptimizer0).getIterationCount();
      doReturn(0, 0, 0, 0, 0).when(univariateRealOptimizer0).getEvaluations();
      doReturn(0.0, 0.0, 0.0, 0.0, 0.0).when(univariateRealOptimizer0).getFunctionValue();
      doReturn(0.0, 0.0, 0.0, 0.0, 0.0).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      JDKRandomGenerator jDKRandomGenerator0 = new JDKRandomGenerator();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 25, jDKRandomGenerator0);
      double[] doubleArray0 = new double[1];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      GoalType goalType0 = GoalType.MINIMIZE;
      multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) polynomialFunction0, goalType0, (double) 25, 25.0);
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaximalIterationCount());
      assertEquals(0, multiStartUnivariateRealOptimizer0.getIterationCount());
      assertEquals(Integer.MAX_VALUE, multiStartUnivariateRealOptimizer0.getMaxEvaluations());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      UnivariateRealOptimizer univariateRealOptimizer0 = mock(UnivariateRealOptimizer.class, new ViolatedAssumptionAnswer());
      doReturn(93, 2021, 2021, 2021, 2021).when(univariateRealOptimizer0).getIterationCount();
      doReturn(93, 2021, 7, 7, 7).when(univariateRealOptimizer0).getEvaluations();
      doReturn((double)2021, (double)93, (double)93, 0.0, (double)93).when(univariateRealOptimizer0).getFunctionValue();
      doReturn(0.0, (double)7, 494.534516136252, (double)7, 494.534516136252).when(univariateRealOptimizer0).optimize(any(org.apache.commons.math.analysis.UnivariateRealFunction.class) , any(org.apache.commons.math.optimization.GoalType.class) , anyDouble() , anyDouble());
      JDKRandomGenerator jDKRandomGenerator0 = new JDKRandomGenerator();
      MultiStartUnivariateRealOptimizer multiStartUnivariateRealOptimizer0 = new MultiStartUnivariateRealOptimizer(univariateRealOptimizer0, 25, jDKRandomGenerator0);
      double[] doubleArray0 = new double[1];
      PolynomialFunction polynomialFunction0 = new PolynomialFunction(doubleArray0);
      GoalType goalType0 = GoalType.MINIMIZE;
      double double0 = multiStartUnivariateRealOptimizer0.optimize((UnivariateRealFunction) polynomialFunction0, goalType0, (double) 25, 25.0);
      assertEquals(2275, multiStartUnivariateRealOptimizer0.getEvaluations());
      assertEquals(494.534516136252, double0, 0.01);
  }
}