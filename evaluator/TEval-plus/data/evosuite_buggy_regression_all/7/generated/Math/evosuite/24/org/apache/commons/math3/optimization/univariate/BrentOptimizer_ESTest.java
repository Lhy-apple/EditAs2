/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:51:15 GMT 2023
 */

package org.apache.commons.math3.optimization.univariate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Signum;
import org.apache.commons.math3.analysis.function.Sin;
import org.apache.commons.math3.analysis.function.Tanh;
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
        brentOptimizer0 = new BrentOptimizer((-298.29363207926), (-298.29363207926));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -298.294 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = null;
      try {
        brentOptimizer0 = new BrentOptimizer(985.8224775492424, 0.0);
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
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.1658105531230376E-8, 2.1658105531230376E-8);
      Signum signum0 = new Signum();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(388, (UnivariateFunction) signum0, goalType0, 2.1658105531230376E-8, (double) 388, 2.1658105531230376E-8);
      assertEquals(387.99998962039325, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(8.304684502905548E-7, 8.304684502905548E-7);
      Tanh tanh0 = new Tanh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(2299, (UnivariateFunction) tanh0, goalType0, (double) 2299, 8.304684502905548E-7, (-2350.02946));
      assertEquals((-2350.026189016333), univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.1658105531230376E-8, 2.1658105531230376E-8);
      Sin sin0 = new Sin();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(380, (UnivariateFunction) sin0, goalType0, (double) 380, (-1267.0), (-1267.0));
      assertEquals((-1261.349422831582), univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.1658105531230376E-8, 2.1658105531230376E-8);
      Sin sin0 = new Sin();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(380, (UnivariateFunction) sin0, goalType0, (double) 380, 2.3887771853026152E-5, 2.3887771853026152E-5);
      assertEquals(190.06636172691327, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.5, 0.5);
      Tanh tanh0 = new Tanh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(685, (UnivariateFunction) tanh0, goalType0, (-1886.7306773717244), (double) 685, (double) 685);
      assertEquals((-1357.1254527897736), univariatePointValuePair0.getPoint(), 0.01);
  }
}
