/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:08:37 GMT 2023
 */

package org.apache.commons.math3.optimization.univariate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Cos;
import org.apache.commons.math3.analysis.function.Cosh;
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
      BrentOptimizer brentOptimizer0 = null;
      try {
        brentOptimizer0 = new BrentOptimizer((-1693.416789), (-1693.416789));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1,693.417 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = null;
      try {
        brentOptimizer0 = new BrentOptimizer(1175.5, 0.0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // 0 is smaller than, or equal to, the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.3472546699189522E-8, 2.3472546699189522E-8);
      Cosh cosh0 = new Cosh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(110, (UnivariateFunction) cosh0, goalType0, (double) 110, (-80.2389807348));
      assertEquals((-3.051946918226542E-9), univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.3472546699189522E-8, 2.3472546699189522E-8);
      Cosh cosh0 = new Cosh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1709, (UnivariateFunction) cosh0, goalType0, 2.3472546699189522E-8, (double) 1709);
      assertEquals(1.0000000000000016, univariatePointValuePair0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.3472546699189522E-8, 2.3472546699189522E-8);
      Cosh cosh0 = new Cosh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(458, (UnivariateFunction) cosh0, goalType0, (-597.1), (double) 458);
      assertEquals(2.348529792757615E-8, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.3472546699189522E-8, 2.3472546699189522E-8);
      Cosh cosh0 = new Cosh();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(2556, (UnivariateFunction) cosh0, goalType0, (double) 2556, 2.3472546699189522E-8);
      assertEquals(1277.9999680942285, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.5, 0.5);
      Cosh cosh0 = new Cosh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(100, (UnivariateFunction) cosh0, goalType0, 0.5, (double) 100);
      assertNotNull(univariatePointValuePair0);
      assertEquals(3.3286468383736264E21, univariatePointValuePair0.getValue(), 0.01);
      assertEquals(50.25, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.3472546699189522E-8, 2.3472546699189522E-8);
      GoalType goalType0 = GoalType.MAXIMIZE;
      Cos cos0 = new Cos();
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(94, (UnivariateFunction) cos0, goalType0, (double) 94, 2.3472546699189522E-8);
      assertEquals(18.84955638736025, univariatePointValuePair0.getPoint(), 0.01);
  }
}
