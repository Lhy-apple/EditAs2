/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:50:56 GMT 2023
 */

package org.apache.commons.math3.optimization.univariate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.function.Atanh;
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
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.22697299641013202, 0.22697299641013202);
      Ulp ulp0 = new Ulp();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1036, (UnivariateFunction) ulp0, goalType0, 0.22697299641013202, (double) 1036, (double) 1036);
      assertEquals(2.2737367544323206E-13, univariatePointValuePair0.getValue(), 0.01);
      assertEquals(1036.0, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = null;
      try {
        brentOptimizer0 = new BrentOptimizer((-4.0746707561835666E-5), (-4.0746707561835666E-5));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -0 is smaller than the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = null;
      try {
        brentOptimizer0 = new BrentOptimizer(2542.9329150574004, (-5.275210583909726E-8));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // -0 is smaller than, or equal to, the minimum (0)
         //
         verifyException("org.apache.commons.math3.optimization.univariate.BrentOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.0734340706476473E-9, 2.0734340706476473E-9);
      Ulp ulp0 = new Ulp();
      GoalType goalType0 = GoalType.MAXIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(991, (UnivariateFunction) ulp0, goalType0, 2.0734340706476473E-9, (double) 991, (double) 991);
      assertEquals(612.4716850173376, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(1.7239809864592146, 1.7239809864592146);
      Atanh atanh0 = new Atanh();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(2591, (UnivariateFunction) atanh0, goalType0, (double) 2591, 8.584676196065558E-8, 1.7239809864592146);
      assertEquals(22.776397150550736, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(0.311917982604859, 0.311917982604859);
      Ulp ulp0 = new Ulp();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(16428, (UnivariateFunction) ulp0, goalType0, 0.311917982604859, (double) 16428, (double) 16428);
      assertEquals(1.4039973276173194, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.0734340706476473E-9, 2.0734340706476473E-9);
      Ulp ulp0 = new Ulp();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(991, (UnivariateFunction) ulp0, goalType0, (-3022.4430245694), (double) 991, (-3022.4430245694));
      assertEquals(0.9333290969856184, univariatePointValuePair0.getPoint(), 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(2.0734E-9, 2.0734E-9);
      Ulp ulp0 = new Ulp();
      GoalType goalType0 = GoalType.MINIMIZE;
      UnivariatePointValuePair univariatePointValuePair0 = brentOptimizer0.optimize(1023, (UnivariateFunction) ulp0, goalType0, (double) 1023, (double) 1023, (double) 1023);
      assertEquals(1023.0, univariatePointValuePair0.getPoint(), 0.01);
      assertEquals(1.1368683772161603E-13, univariatePointValuePair0.getValue(), 0.01);
      assertNotNull(univariatePointValuePair0);
  }
}