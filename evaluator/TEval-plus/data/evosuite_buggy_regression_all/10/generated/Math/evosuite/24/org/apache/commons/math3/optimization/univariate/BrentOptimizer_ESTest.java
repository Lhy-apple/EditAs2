/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:20:58 GMT 2023
 */

package org.apache.commons.math3.optimization.univariate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.HarmonicOscillator;
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
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.008333333333329196, 0.008333333333329196);
      HarmonicOscillator harmonicOscillator0 = new HarmonicOscillator(0.008333333333329196, 0.008333333333329196, 0.008333333333329196);
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(3145, (UnivariateFunction) harmonicOscillator0, goalType0, 0.008333333333329196, (double) 3145, (double) 3145);
      assertEquals(3038.8615716476384, univariatePointValuePair0.getPoint(), 0.01);
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
        brentOptimizer0 = new BrentOptimizer(0.008333333333329196, (-1.0));
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
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.008333333333329196, 0.008333333333329196);
      HarmonicOscillator harmonicOscillator0 = new HarmonicOscillator(0.008333333333329196, 0.008333333333329196, 0.008333333333329196);
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(3145, (UnivariateFunction) harmonicOscillator0, goalType0, 0.008333333333329196, (double) 3145, (double) 3145);
      assertEquals(1866.0824583952824, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.10508785561343062, 0.10508785561343062);
      HarmonicOscillator harmonicOscillator0 = new HarmonicOscillator(0.10508785561343062, 0.10508785561343062, 0.10508785561343062);
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(3154, (UnivariateFunction) harmonicOscillator0, goalType0, (double) 3154, (double) 3154, 0.10508785561343062);
      assertEquals(0.31358226467850064, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.10508785561343062, 0.10508785561343062);
      HarmonicOscillator harmonicOscillator0 = new HarmonicOscillator(0.10508785561343062, 0.10508785561343062, 0.10508785561343062);
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(3154, (UnivariateFunction) harmonicOscillator0, goalType0, (double) 3154, (double) 3154, 0.10508785561343062);
      assertEquals(0.03766427762519545, univariatePointValuePair0.getValue(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.10508785561343062, 0.10508785561343062);
      HarmonicOscillator harmonicOscillator0 = new HarmonicOscillator(0.10508785561343062, 0.10508785561343062, 0.10508785561343062);
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(3143, (UnivariateFunction) harmonicOscillator0, goalType0, 0.10508785561343062, (double) 3143, (double) 3143);
      assertEquals(0.09602066938307714, univariatePointValuePair0.getValue(), 0.01);
  }
}