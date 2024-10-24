/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:51:28 GMT 2023
 */

package org.apache.commons.math.estimation;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.estimation.EstimatedParameter;
import org.apache.commons.math.estimation.GaussNewtonEstimator;
import org.apache.commons.math.estimation.LevenbergMarquardtEstimator;
import org.apache.commons.math.estimation.SimpleEstimationProblem;
import org.apache.commons.math.estimation.WeightedMeasurement;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class AbstractEstimator_ESTest extends AbstractEstimator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      LevenbergMarquardtEstimator levenbergMarquardtEstimator0 = new LevenbergMarquardtEstimator();
      int int0 = levenbergMarquardtEstimator0.getCostEvaluations();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      GaussNewtonEstimator gaussNewtonEstimator0 = new GaussNewtonEstimator(1, 1, 1.0);
      int int0 = gaussNewtonEstimator0.getJacobianEvaluations();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      WeightedMeasurement weightedMeasurement0 = mock(WeightedMeasurement.class, new ViolatedAssumptionAnswer());
      doReturn(2176.175, 2176.175).when(weightedMeasurement0).getResidual();
      doReturn(2176.175, 2176.175, 2176.175, 2176.175, 2176.175).when(weightedMeasurement0).getWeight();
      SimpleEstimationProblem simpleEstimationProblem0 = new SimpleEstimationProblem();
      simpleEstimationProblem0.addMeasurement(weightedMeasurement0);
      LevenbergMarquardtEstimator levenbergMarquardtEstimator0 = new LevenbergMarquardtEstimator();
      levenbergMarquardtEstimator0.estimate(simpleEstimationProblem0);
      levenbergMarquardtEstimator0.cols = 7;
      // Undeclared exception!
      try { 
        levenbergMarquardtEstimator0.guessParametersErrors(simpleEstimationProblem0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.math.estimation.AbstractEstimator", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      GaussNewtonEstimator gaussNewtonEstimator0 = new GaussNewtonEstimator((-1192), (-1192), (-1192));
      try { 
        gaussNewtonEstimator0.updateResidualsAndCost();
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // maximal number of evaluations exceeded (-1,192)
         //
         verifyException("org.apache.commons.math.estimation.AbstractEstimator", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SimpleEstimationProblem simpleEstimationProblem0 = new SimpleEstimationProblem();
      LevenbergMarquardtEstimator levenbergMarquardtEstimator0 = new LevenbergMarquardtEstimator();
      double double0 = levenbergMarquardtEstimator0.getRMS(simpleEstimationProblem0);
      assertEquals(Double.NaN, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SimpleEstimationProblem simpleEstimationProblem0 = new SimpleEstimationProblem();
      simpleEstimationProblem0.addMeasurement((WeightedMeasurement) null);
      LevenbergMarquardtEstimator levenbergMarquardtEstimator0 = new LevenbergMarquardtEstimator();
      // Undeclared exception!
      try { 
        levenbergMarquardtEstimator0.getRMS(simpleEstimationProblem0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math.estimation.AbstractEstimator", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      LevenbergMarquardtEstimator levenbergMarquardtEstimator0 = new LevenbergMarquardtEstimator();
      SimpleEstimationProblem simpleEstimationProblem0 = new SimpleEstimationProblem();
      levenbergMarquardtEstimator0.estimate(simpleEstimationProblem0);
      simpleEstimationProblem0.addParameter((EstimatedParameter) null);
      try { 
        levenbergMarquardtEstimator0.getCovariances(simpleEstimationProblem0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // unable to compute covariances: singular problem
         //
         verifyException("org.apache.commons.math.estimation.AbstractEstimator", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      LevenbergMarquardtEstimator levenbergMarquardtEstimator0 = new LevenbergMarquardtEstimator();
      SimpleEstimationProblem simpleEstimationProblem0 = new SimpleEstimationProblem();
      levenbergMarquardtEstimator0.estimate(simpleEstimationProblem0);
      WeightedMeasurement weightedMeasurement0 = mock(WeightedMeasurement.class, new ViolatedAssumptionAnswer());
      simpleEstimationProblem0.addMeasurement(weightedMeasurement0);
      simpleEstimationProblem0.addParameter((EstimatedParameter) null);
      // Undeclared exception!
      try { 
        levenbergMarquardtEstimator0.getCovariances(simpleEstimationProblem0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 0
         //
         verifyException("org.apache.commons.math.estimation.AbstractEstimator", e);
      }
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      SimpleEstimationProblem simpleEstimationProblem0 = new SimpleEstimationProblem();
      LevenbergMarquardtEstimator levenbergMarquardtEstimator0 = new LevenbergMarquardtEstimator();
      try { 
        levenbergMarquardtEstimator0.guessParametersErrors(simpleEstimationProblem0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // no degrees of freedom (0 measurements, 0 parameters)
         //
         verifyException("org.apache.commons.math.estimation.AbstractEstimator", e);
      }
  }
}
