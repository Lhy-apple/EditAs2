/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:05:49 GMT 2023
 */

package org.apache.commons.math.analysis.solvers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.UnivariateRealFunction;
import org.apache.commons.math.analysis.function.Sqrt;
import org.apache.commons.math.analysis.function.Tan;
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
      RegulaFalsiSolver regulaFalsiSolver0 = new RegulaFalsiSolver();
      Tan tan0 = new Tan();
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      regulaFalsiSolver0.solve(1778, (UnivariateRealFunction) tan0, 1.0, (double) 1778, (double) 1778, allowedSolution0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(365.098355884, 365.098355884);
      Sqrt sqrt0 = new Sqrt();
      // Undeclared exception!
      try { 
        pegasusSolver0.solve(0, (UnivariateRealFunction) sqrt0, 365.098355884, (double) 0, 365.098355884);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // illegal state: maximal count (0) exceeded: evaluations
         //
         verifyException("org.apache.commons.math.analysis.solvers.BaseAbstractUnivariateRealSolver", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1749, (-4963.48535), 1749);
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = illinoisSolver0.solve(1749, (UnivariateRealFunction) tan0, (-4963.48535), (double) 1749, allowedSolution0);
      assertEquals((-1607.242675), illinoisSolver0.getStartValue(), 0.01);
      assertEquals((-4963.48535), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Tan tan0 = new Tan();
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = pegasusSolver0.solve(1741, (UnivariateRealFunction) tan0, 0.0, (double) 1741, allowedSolution0);
      assertEquals(1741.0, pegasusSolver0.getMax(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = illinoisSolver0.solve(1749, (UnivariateRealFunction) tan0, (double) 1749, 0.0, allowedSolution0);
      assertEquals(1749.0, illinoisSolver0.getMin(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = illinoisSolver0.solve(1755, (UnivariateRealFunction) tan0, 5.298012724256067E-18, (double) 1755, allowedSolution0);
      assertEquals(877.5, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(0.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Tan tan0 = new Tan();
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = pegasusSolver0.solve(1741, (UnivariateRealFunction) tan0, (-10.42055630562592), (double) 1741, allowedSolution0);
      assertEquals(865.289721847187, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(1583.3626974092558, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = illinoisSolver0.solve(1753, (UnivariateRealFunction) tan0, 1013.0916210815689, (double) 1753, allowedSolution0);
      assertEquals(1383.0458105407845, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1566.0839373588567, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1765, 1012.2058555128416, 1765);
      AllowedSolution allowedSolution0 = AllowedSolution.ANY_SIDE;
      double double0 = illinoisSolver0.solve(1765, (UnivariateRealFunction) tan0, 1012.2058555128416, (double) 1765, allowedSolution0);
      assertEquals(1388.6029277564207, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1403.9973273777928, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1753, 1012.2058555128416, 1753);
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = illinoisSolver0.solve(1753, (UnivariateRealFunction) tan0, 1012.2058555128416, (double) 1753, allowedSolution0);
      assertEquals(1382.6029277564207, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1753.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1753, 1012.2058555128416, 1753);
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = illinoisSolver0.solve(1753, (UnivariateRealFunction) tan0, 1012.2058555128416, (double) 1753, allowedSolution0);
      assertEquals(1382.6029277564207, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1753.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1749, 1012.676112, 1749);
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = illinoisSolver0.solve(1749, (UnivariateRealFunction) tan0, 1012.676112, (double) 1749, allowedSolution0);
      assertEquals(1380.838056, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1012.676112, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1753, 1012.2058555128416, 1753);
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = illinoisSolver0.solve(1753, (UnivariateRealFunction) tan0, 1012.2058555128416, (double) 1753, allowedSolution0);
      assertEquals(1382.6029277564207, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1743.948804731383, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1765, 1012.2058555128416, 1765);
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = illinoisSolver0.solve(1765, (UnivariateRealFunction) tan0, 1012.2058555128416, (double) 1765, allowedSolution0);
      assertEquals(1388.6029277564207, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1403.9973273777928, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1753, 1012.676112, 1753);
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = illinoisSolver0.solve(1753, (UnivariateRealFunction) tan0, 1012.676112, (double) 1753, allowedSolution0);
      assertEquals(1382.838056, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1749.6003660339422, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1753, 1012.9206588235013, 1753);
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = illinoisSolver0.solve(1753, (UnivariateRealFunction) tan0, 1012.9206588235013, (double) 1753, allowedSolution0);
      assertEquals(1382.9603294117505, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1751.4073143174892, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1012.676112, 1749);
      AllowedSolution allowedSolution0 = AllowedSolution.ANY_SIDE;
      double double0 = illinoisSolver0.solve(1749, (UnivariateRealFunction) tan0, 1012.676112, (double) 1749, allowedSolution0);
      assertEquals(1380.838056, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1465.856965381576, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1749);
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = illinoisSolver0.solve(1749, (UnivariateRealFunction) tan0, (double) 1749, 2686.515074452, allowedSolution0);
      assertEquals(2217.757537226, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(2412.283909520423, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Tan tan0 = new Tan();
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(1753);
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = illinoisSolver0.solve(1753, (UnivariateRealFunction) tan0, 1012.4450762752758, (double) 1753, allowedSolution0);
      assertEquals(1382.722538137638, illinoisSolver0.getStartValue(), 0.01);
      assertEquals(1747.4076141297878, double0, 0.01);
  }
}
