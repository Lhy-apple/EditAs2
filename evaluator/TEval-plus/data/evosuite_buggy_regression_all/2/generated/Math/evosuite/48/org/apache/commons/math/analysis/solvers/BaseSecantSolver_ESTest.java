/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:28:34 GMT 2023
 */

package org.apache.commons.math.analysis.solvers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.UnivariateRealFunction;
import org.apache.commons.math.analysis.function.Cos;
import org.apache.commons.math.analysis.function.Gaussian;
import org.apache.commons.math.analysis.function.Logit;
import org.apache.commons.math.analysis.function.Minus;
import org.apache.commons.math.analysis.solvers.AllowedSolution;
import org.apache.commons.math.analysis.solvers.IllinoisSolver;
import org.apache.commons.math.analysis.solvers.PegasusSolver;
import org.apache.commons.math.analysis.solvers.RegulaFalsiSolver;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class BaseSecantSolver_ESTest extends BaseSecantSolver_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      Gaussian gaussian0 = new Gaussian();
      pegasusSolver0.solve(665, (UnivariateRealFunction) gaussian0, (double) 665, (double) 665, (double) 665, allowedSolution0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(501.9457236, 501.9457236);
      assertEquals(0.0, pegasusSolver0.getMax(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Cos cos0 = new Cos();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(665);
      AllowedSolution allowedSolution0 = AllowedSolution.ANY_SIDE;
      double double0 = illinoisSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 857.9062732623711, allowedSolution0);
      assertEquals(665.0, illinoisSolver0.getMin(), 0.01);
      assertEquals(732.8402362226996, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Gaussian gaussian0 = new Gaussian(665, 665, 0.036);
      AllowedSolution allowedSolution0 = AllowedSolution.ANY_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) gaussian0, (double) 665, 0.036, (double) 665, allowedSolution0);
      assertEquals(0.036, pegasusSolver0.getMax(), 0.01);
      assertEquals(0.036, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(666, 666, 666);
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      Minus minus0 = new Minus();
      double double0 = pegasusSolver0.solve(566, (UnivariateRealFunction) minus0, (-2936.41874107371), (double) 666, (double) 666, allowedSolution0);
      assertEquals(666.0, pegasusSolver0.getMax(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = pegasusSolver0.solve(648, (UnivariateRealFunction) cos0, (-1510.0), (double) 648, (-1510.0), allowedSolution0);
      assertEquals((-1510.0), pegasusSolver0.getStartValue(), 0.01);
      assertEquals((-755.5530331883449), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Logit logit0 = new Logit();
      UnivariateRealFunction univariateRealFunction0 = logit0.derivative();
      RegulaFalsiSolver regulaFalsiSolver0 = new RegulaFalsiSolver(5093.363427407);
      // Undeclared exception!
      try { 
        regulaFalsiSolver0.solve(5, univariateRealFunction0, (-1.0), 1.0, (double) 0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: maximal count (5) exceeded: evaluations
         //
         verifyException("org.apache.commons.math.analysis.solvers.BaseAbstractUnivariateRealSolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(857.9062732623711, 857.9062732623711, 665);
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 857.9062732623711, 857.9062732623711, allowedSolution0);
      assertEquals(665.0, pegasusSolver0.getMin(), 0.01);
      assertEquals(665.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(858.126902910437, 858.126902910437, 665);
      AllowedSolution allowedSolution0 = AllowedSolution.ANY_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 858.126902910437, 858.126902910437, allowedSolution0);
      assertEquals(858.126902910437, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(736.6559412239131, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(858.12690291, 858.12690291, 698);
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = pegasusSolver0.solve(698, (UnivariateRealFunction) cos0, (double) 698, 858.12690291, 858.12690291, allowedSolution0);
      assertEquals(698.0, pegasusSolver0.getMin(), 0.01);
      assertEquals(698.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(858.126902910437, 858.126902910437, 665);
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 858.126902910437, 858.126902910437, allowedSolution0);
      assertEquals(858.126902910437, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(858.126902910437, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(858.126902910437, 858.126902910437, 665);
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 858.126902910437, 858.126902910437, allowedSolution0);
      assertEquals(858.126902910437, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(858.126902910437, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(2157.7144966994706, 2157.7144966994706, 698);
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = pegasusSolver0.solve(698, (UnivariateRealFunction) cos0, (double) 698, 2157.7144966994706, 2157.7144966994706, allowedSolution0);
      assertEquals(2157.7144966994706, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(1426.296770119619, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(858.1381663019363, 858.1381663019363, 665);
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 858.1381663019363, 858.1381663019363, allowedSolution0);
      assertEquals(665.0, pegasusSolver0.getMin(), 0.01);
      assertEquals(736.9231598408758, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(857.9062732623711, 857.9062732623711, 665);
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 857.9062732623711, 857.9062732623711, allowedSolution0);
      assertEquals(665.0, pegasusSolver0.getMin(), 0.01);
      assertEquals(732.8402362226996, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver(858.126902910437, 858.126902910437, 665);
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 858.126902910437, 858.126902910437, allowedSolution0);
      assertEquals(665.0, pegasusSolver0.getMin(), 0.01);
      assertEquals(736.6559412239131, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 1341.98805507, 1341.98805507, allowedSolution0);
      assertEquals(1341.98805507, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(692.7211801165494, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 857.9062732623711, 857.9062732623711, allowedSolution0);
      assertEquals(857.9062732623711, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(673.8716241950107, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Cos cos0 = new Cos();
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = pegasusSolver0.solve(665, (UnivariateRealFunction) cos0, (double) 665, 858.126902910437, 858.126902910437, allowedSolution0);
      assertEquals(858.126902910437, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(856.0839981032186, double0, 0.01);
  }
}