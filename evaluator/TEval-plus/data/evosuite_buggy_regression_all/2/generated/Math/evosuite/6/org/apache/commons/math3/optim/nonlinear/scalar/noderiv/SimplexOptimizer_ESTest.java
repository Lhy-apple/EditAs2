/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:24:32 GMT 2023
 */

package org.apache.commons.math3.optim.nonlinear.scalar.noderiv;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.SimpleBounds;
import org.apache.commons.math3.optim.SimpleValueChecker;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.MultiDirectionalSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class SimplexOptimizer_ESTest extends SimplexOptimizer_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      SimplexOptimizer simplexOptimizer0 = new SimplexOptimizer((-1587.38), (-1587.38));
      OptimizationData[] optimizationDataArray0 = new OptimizationData[2];
      // Undeclared exception!
      try { 
        simplexOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // null is not allowed
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      SimplexOptimizer simplexOptimizer0 = new SimplexOptimizer((-1587.38), (-1587.38));
      OptimizationData[] optimizationDataArray0 = new OptimizationData[2];
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = (-1587.38);
      doubleArray0[1] = (-1587.38);
      MultiDirectionalSimplex multiDirectionalSimplex0 = new MultiDirectionalSimplex(doubleArray0);
      optimizationDataArray0[0] = (OptimizationData) multiDirectionalSimplex0;
      InitialGuess initialGuess0 = new InitialGuess(doubleArray0);
      optimizationDataArray0[1] = (OptimizationData) initialGuess0;
      // Undeclared exception!
      try { 
        simplexOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: maximal count (0) exceeded: evaluations
         //
         verifyException("org.apache.commons.math3.optim.BaseOptimizer$MaxEvalCallback", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      double[] doubleArray0 = new double[3];
      SimpleBounds simpleBounds0 = new SimpleBounds(doubleArray0, doubleArray0);
      MultiDirectionalSimplex multiDirectionalSimplex0 = new MultiDirectionalSimplex(1);
      SimpleValueChecker simpleValueChecker0 = new SimpleValueChecker(1, (-2089.012448703275));
      SimplexOptimizer simplexOptimizer0 = new SimplexOptimizer(simpleValueChecker0);
      OptimizationData[] optimizationDataArray0 = new OptimizationData[2];
      optimizationDataArray0[0] = (OptimizationData) simpleBounds0;
      optimizationDataArray0[1] = (OptimizationData) multiDirectionalSimplex0;
      // Undeclared exception!
      try { 
        simplexOptimizer0.optimize(optimizationDataArray0);
        fail("Expecting exception: MathUnsupportedOperationException");
      
      } catch(MathUnsupportedOperationException e) {
         //
         // constraint
         //
         verifyException("org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer", e);
      }
  }
}