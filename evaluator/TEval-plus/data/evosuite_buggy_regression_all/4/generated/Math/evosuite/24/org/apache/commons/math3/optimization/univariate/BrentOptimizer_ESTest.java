/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:08:49 GMT 2023
 */

package org.apache.commons.math3.optimization.univariate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Acos;
import org.apache.commons.math3.analysis.function.Tan;
import org.apache.commons.math3.analysis.function.Tanh;
import org.apache.commons.math3.analysis.function.Ulp;
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
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.11140543538534475, 0.11140543538534475);
      Tan tan0 = new Tan();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(383, (UnivariateFunction) tan0, goalType0, 0.11140543538534475, (double) 383, (-1.0));
      assertEquals(61.266317205299856, univariatePointValuePair0.getPoint(), 0.01);
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
        brentOptimizer0 = new BrentOptimizer(1.0, (-1.0));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -1 is smaller than, or equal to, the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.276398261546674, 0.276398261546674);
      Tanh tanh0 = new Tanh();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(2602, (UnivariateFunction) tanh0, goalType0, 0.276398261546674, (double) 2602, (double) 2602);
      assertEquals(27.629639779207817, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.276398261546674, 0.276398261546674);
      Tanh tanh0 = new Tanh();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(2623, (UnivariateFunction) tanh0, goalType0, (double) 2623, 0.276398261546674, 0.276398261546674);
      assertEquals(2069.5843993053463, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.276398261546674, 0.276398261546674);
      Tanh tanh0 = new Tanh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(2623, (UnivariateFunction) tanh0, goalType0, (double) 2623, 0.276398261546674, 0.276398261546674);
      assertEquals(0.6232572597589859, univariatePointValuePair0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.276398261546674, 0.276398261546674);
      GoalType goalType0 = GoalType.MINIMIZE;
      Ulp ulp0 = new Ulp();
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(57, (UnivariateFunction) ulp0, goalType0, 0.276398261546674, (double) 57, 2083.05);
      assertEquals(1.3785637987010726, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.276398261546674, 0.276398261546674);
      Acos acos0 = new Acos();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1093, (UnivariateFunction) acos0, goalType0, 0.276398261546674, 1754.9322612255144, (double) 1093);
      assertEquals(1395.3796981320613, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.276398261546674, 0.276398261546674);
      Tanh tanh0 = new Tanh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1969126180, (UnivariateFunction) tanh0, goalType0, 0.276398261546674, (-44.97325397855559), (-44.97325397855559));
      assertEquals((-35.619131941982175), univariatePointValuePair0.getPoint(), 0.01);
  }
}