/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:05:39 GMT 2023
 */

package org.apache.commons.math.analysis.solvers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.analysis.UnivariateRealFunction;
import org.apache.commons.math.analysis.function.Cbrt;
import org.apache.commons.math.analysis.function.Gaussian;
import org.apache.commons.math.analysis.function.HarmonicOscillator;
import org.apache.commons.math.analysis.function.Log;
import org.apache.commons.math.analysis.function.Log10;
import org.apache.commons.math.analysis.function.Sinc;
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
      Sinc sinc0 = new Sinc();
      pegasusSolver0.solve(156, (UnivariateRealFunction) sinc0, (-33.557279), (double) 156, (double) 156);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(1.039134509928639, 1.039134509928639, 1.039134509928639);
      Sinc sinc0 = new Sinc();
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = pegasusSolver0.solve(144, (UnivariateRealFunction) sinc0, 1.039134509928639, (double) 144, allowedSolution0);
      assertEquals(144.0, pegasusSolver0.getMax(), 0.01);
      assertEquals(1.039134509928639, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(2780);
      Gaussian gaussian0 = new Gaussian();
      double double0 = pegasusSolver0.solve(2780, (UnivariateRealFunction) gaussian0, (double) 2780, (double) 2780, (double) 2780);
      assertEquals(2780.0, pegasusSolver0.getMin(), 0.01);
      assertEquals(2780.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(2780);
      Gaussian gaussian0 = new Gaussian();
      double double0 = pegasusSolver0.solve(2780, (UnivariateRealFunction) gaussian0, 0.005852441691290568, (double) 2780, (double) 2780);
      assertEquals(2780.0, pegasusSolver0.getMax(), 0.01);
      assertEquals(2780.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Log10 log10_0 = new Log10();
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      double double0 = pegasusSolver0.solve(2774, (UnivariateRealFunction) log10_0, 0.07692307692307693, (double) 2774, (double) 2774);
      assertEquals(0.07692307692307693, pegasusSolver0.getMin(), 0.01);
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      IllinoisSolver illinoisSolver0 = new IllinoisSolver(0.0, 2811);
      Log10 log10_0 = new Log10();
      // Undeclared exception!
      illinoisSolver0.solve(2811, (UnivariateRealFunction) log10_0, 0.0, (double) 2811, (double) 2811);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      RegulaFalsiSolver regulaFalsiSolver0 = new RegulaFalsiSolver(510.569889979);
      Cbrt cbrt0 = new Cbrt();
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = regulaFalsiSolver0.solve(1832, (UnivariateRealFunction) cbrt0, (-0.1666666567325592), (double) 1832, allowedSolution0);
      assertEquals(1832.0, regulaFalsiSolver0.getMax(), 0.01);
      assertEquals(78.68908131288481, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(1.176342548272881E-8, 1.176342548272881E-8);
      Sinc sinc0 = new Sinc();
      double double0 = pegasusSolver0.solve(224, (UnivariateRealFunction) sinc0, 1.176342548272881E-8, (double) 224, 1.176342548272881E-8);
      assertEquals(1.176342548272881E-8, pegasusSolver0.getMin(), 0.01);
      assertEquals(223.05307840487532, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Sinc sinc0 = new Sinc();
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = pegasusSolver0.solve(167, (UnivariateRealFunction) sinc0, (-33.557279), 1493.35, allowedSolution0);
      assertEquals((-33.557279), pegasusSolver0.getMin(), 0.01);
      assertEquals(1492.2565104551536, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(1.039134509928639, 1.039134509928639, 1.039134509928639);
      Sinc sinc0 = new Sinc();
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = pegasusSolver0.solve(144, (UnivariateRealFunction) sinc0, 1.039134509928639, (double) 144, allowedSolution0);
      assertEquals(144.0, pegasusSolver0.getMax(), 0.01);
      assertEquals(143.4147311605369, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Sinc sinc0 = new Sinc();
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = pegasusSolver0.solve(156, (UnivariateRealFunction) sinc0, (-1303.1924), (double) 156, allowedSolution0);
      assertEquals((-573.5962), pegasusSolver0.getStartValue(), 0.01);
      assertEquals((-1206.3715789784808), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver(1.039134509928639, 1.039134509928639, 1.039134509928639);
      Sinc sinc0 = new Sinc();
      AllowedSolution allowedSolution0 = AllowedSolution.RIGHT_SIDE;
      double double0 = pegasusSolver0.solve(144, (UnivariateRealFunction) sinc0, 1.039134509928639, (double) 144, allowedSolution0);
      assertEquals(72.51956725496433, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(143.4147311605369, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      HarmonicOscillator harmonicOscillator0 = new HarmonicOscillator(1.0E-6, 11.457673585394211, 1.0E-6);
      UnivariateRealFunction univariateRealFunction0 = harmonicOscillator0.derivative();
      double double0 = pegasusSolver0.solve(16, univariateRealFunction0, 11.457673585394211, (double) 16, allowedSolution0);
      assertEquals(16.0, pegasusSolver0.getMax(), 0.01);
      assertEquals(12.61279263964454, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Sinc sinc0 = new Sinc();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = pegasusSolver0.solve(154, (UnivariateRealFunction) sinc0, (-3141.8932426826464), (double) 154, allowedSolution0);
      assertEquals((-1493.9466213413232), pegasusSolver0.getStartValue(), 0.01);
      assertEquals((-1413.716694115407), double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Sinc sinc0 = new Sinc();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = pegasusSolver0.solve(144, (UnivariateRealFunction) sinc0, 1.039134509928639, (double) 144, allowedSolution0);
      assertEquals(72.51956725496433, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(141.37166941154018, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Log10 log10_0 = new Log10();
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      AllowedSolution allowedSolution0 = AllowedSolution.LEFT_SIDE;
      double double0 = pegasusSolver0.solve(773492274, (UnivariateRealFunction) log10_0, 0.2508252982949073, (double) 2764, allowedSolution0);
      assertEquals(1382.1254126491476, pegasusSolver0.getStartValue(), 0.01);
      assertEquals(0.9999999999076127, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Log log0 = new Log();
      AllowedSolution allowedSolution0 = AllowedSolution.BELOW_SIDE;
      double double0 = pegasusSolver0.solve(1589456320, (UnivariateRealFunction) log0, 1.0E-6, 3130.4, allowedSolution0);
      assertEquals(3130.4, pegasusSolver0.getMax(), 0.01);
      assertEquals(0.9999999999833448, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      PegasusSolver pegasusSolver0 = new PegasusSolver();
      Log log0 = new Log();
      AllowedSolution allowedSolution0 = AllowedSolution.ABOVE_SIDE;
      double double0 = pegasusSolver0.solve(1589456320, (UnivariateRealFunction) log0, 1.0E-6, 3130.4, allowedSolution0);
      assertEquals(3130.4, pegasusSolver0.getMax(), 0.01);
      assertEquals(1.0000000922511523, double0, 0.01);
  }
}