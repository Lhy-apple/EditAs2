/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:20:54 GMT 2023
 */

package org.apache.commons.math3.optimization.univariate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Exp;
import org.apache.commons.math3.analysis.function.Log10;
import org.apache.commons.math3.analysis.function.Sinc;
import org.apache.commons.math3.optimization.GoalType;
import org.apache.commons.math3.optimization.univariate.BrentOptimizer;
import org.apache.commons.math3.optimization.univariate.UnivariatePointValuePair;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BrentOptimizer_ESTest extends BrentOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.0588235, 0.0588235);
      Log10 log10_0 = new Log10();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(344777109, (UnivariateFunction) log10_0, goalType0, 0.0588235, (double) 344777109);
      assertEquals(3.219688410467906E8, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = null;
      try {
        brentOptimizer0 = new BrentOptimizer(0.0, 0.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 0 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = null;
      try {
        brentOptimizer0 = new BrentOptimizer(2533.5736433, (-1.5291605014052314));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1.529 is smaller than, or equal to, the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.058823529411764705, 0.058823529411764705);
      Log10 log10_0 = new Log10();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1236, (UnivariateFunction) log10_0, goalType0, (double) 1236, (-508.1275361527096));
      assertEquals((-0.0815296193861505), univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.058823529411764705, 0.058823529411764705);
      GoalType goalType0 = GoalType.MAXIMIZE;
      Sinc sinc0 = new Sinc(true);
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(344777109, (UnivariateFunction) sinc0, goalType0, (double) 344777109, 0.058823529411764705);
      assertEquals(1.8252905779584774E8, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.058823529411764705, 0.058823529411764705);
      Log10 log10_0 = new Log10();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1236, (UnivariateFunction) log10_0, goalType0, (double) 1236, 0.058823529411764705);
      assertEquals(1154.3002820856998, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.1, 0.1);
      Exp exp0 = new Exp();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(52, (UnivariateFunction) exp0, goalType0, 0.1, (double) 52);
      assertEquals(0.3307529627594433, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.18090582773656855, 0.18090582773656855);
      Log10 log10_0 = new Log10();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1230, (UnivariateFunction) log10_0, goalType0, (double) 1230, (-1461.85));
      assertEquals((-144.57467752775324), univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.058823529411764705, 0.058823529411764705);
      Log10 log10_0 = new Log10();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(5314, (UnivariateFunction) log10_0, goalType0, 0.058823529411764705, 0.058823529411764705);
      assertNotNull(univariatePointValuePair0);
      assertEquals((-1.2304489213782739), univariatePointValuePair0.getValue(), 0.01);
      assertEquals(0.058823529411764705, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.058823529411764705, 0.058823529411764705);
      Log10 log10_0 = new Log10();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(2144325117, (UnivariateFunction) log10_0, goalType0, 0.058823529411764705, (-834.9));
      assertEquals((-442.03356401384076), univariatePointValuePair0.getPoint(), 0.01);
  }
}