/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 18:03:18 GMT 2023
 */

package org.apache.commons.math3.optim;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.MaxIter;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.linear.SimplexSolver;
import org.apache.commons.math3.optim.univariate.BrentOptimizer;
import org.apache.commons.math3.optim.univariate.MultiStartUnivariateOptimizer;
import org.apache.commons.math3.random.MersenneTwister;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BaseOptimizer_ESTest extends BaseOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimplexSolver simplexSolver0 = new SimplexSolver();
      int int0 = simplexSolver0.getIterations();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimplexSolver simplexSolver0 = new SimplexSolver();
      int int0 = simplexSolver0.getEvaluations();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      SimplexSolver simplexSolver0 = new SimplexSolver();
      int int0 = simplexSolver0.getMaxIterations();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(3050.2388277683, 1644.7548471622);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[1];
      try { 
        brentOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: maximal count (0) exceeded: evaluations
         //
         verifyException("org.apache.commons.math3.optim.BaseOptimizer$MaxEvalCallback", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SimplexSolver simplexSolver0 = new SimplexSolver();
      int int0 = simplexSolver0.getMaxEvaluations();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(3542.65009, 3542.65009);
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateOptimizer multiStartUnivariateOptimizer0 = new MultiStartUnivariateOptimizer(brentOptimizer0, 1, mersenneTwister0);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[5];
      MaxEval maxEval0 = MaxEval.unlimited();
      optimizationDataArray0[0] = (OptimizationData) maxEval0;
      // Undeclared exception!
      try { 
        multiStartUnivariateOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state
         //
         verifyException("org.apache.commons.math3.optim.univariate.MultiStartUnivariateOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      BrentOptimizer brentOptimizer0 = new BrentOptimizer(3542.65009, 3542.65009);
      MersenneTwister mersenneTwister0 = new MersenneTwister();
      MultiStartUnivariateOptimizer multiStartUnivariateOptimizer0 = new MultiStartUnivariateOptimizer(brentOptimizer0, 1, mersenneTwister0);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[5];
      MaxIter maxIter0 = MaxIter.unlimited();
      optimizationDataArray0[3] = (OptimizationData) maxIter0;
      // Undeclared exception!
      try { 
        multiStartUnivariateOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state
         //
         verifyException("org.apache.commons.math3.optim.univariate.MultiStartUnivariateOptimizer", e);
      }
  }
}
