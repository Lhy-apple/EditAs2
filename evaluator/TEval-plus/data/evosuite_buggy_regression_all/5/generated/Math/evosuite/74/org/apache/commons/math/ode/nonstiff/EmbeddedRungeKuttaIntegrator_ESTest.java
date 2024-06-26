/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:34:37 GMT 2023
 */

package org.apache.commons.math.ode.nonstiff;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.ode.FirstOrderConverter;
import org.apache.commons.math.ode.SecondOrderDifferentialEquations;
import org.apache.commons.math.ode.events.EventHandler;
import org.apache.commons.math.ode.nonstiff.DormandPrince54Integrator;
import org.apache.commons.math.ode.nonstiff.DormandPrince853Integrator;
import org.apache.commons.math.ode.nonstiff.HighamHall54Integrator;
import org.apache.commons.math.ode.sampling.FixedStepHandler;
import org.apache.commons.math.ode.sampling.StepNormalizer;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class EmbeddedRungeKuttaIntegrator_ESTest extends EmbeddedRungeKuttaIntegrator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      double[] doubleArray0 = new double[0];
      HighamHall54Integrator highamHall54Integrator0 = new HighamHall54Integrator((-4156.541616), (-4156.541616), doubleArray0, doubleArray0);
      SecondOrderDifferentialEquations secondOrderDifferentialEquations0 = mock(SecondOrderDifferentialEquations.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(secondOrderDifferentialEquations0).getDimension();
      FirstOrderConverter firstOrderConverter0 = new FirstOrderConverter(secondOrderDifferentialEquations0);
      // Undeclared exception!
      highamHall54Integrator0.integrate(firstOrderConverter0, (-4156.541616), doubleArray0, 1.1385160815281895E-19, doubleArray0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      HighamHall54Integrator highamHall54Integrator0 = new HighamHall54Integrator(15.279233632882423, 54.1, 2760.50491165, 15.279233632882423);
      double double0 = highamHall54Integrator0.getSafety();
      assertEquals(0.2, highamHall54Integrator0.getMinReduction(), 0.01);
      assertEquals(10.0, highamHall54Integrator0.getMaxGrowth(), 0.01);
      assertEquals(0.9, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      double[] doubleArray0 = new double[1];
      DormandPrince853Integrator dormandPrince853Integrator0 = new DormandPrince853Integrator((-1.1270175653862835), (-1.1270175653862835), doubleArray0, doubleArray0);
      double double0 = dormandPrince853Integrator0.getMinReduction();
      assertEquals(0.2, double0, 0.01);
      assertEquals(0.9, dormandPrince853Integrator0.getSafety(), 0.01);
      assertEquals(10.0, dormandPrince853Integrator0.getMaxGrowth(), 0.01);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      double[] doubleArray0 = new double[6];
      DormandPrince54Integrator dormandPrince54Integrator0 = new DormandPrince54Integrator((-1.0), 342.5, doubleArray0, doubleArray0);
      double double0 = dormandPrince54Integrator0.getMaxGrowth();
      assertEquals(0.9, dormandPrince54Integrator0.getSafety(), 0.01);
      assertEquals(0.2, dormandPrince54Integrator0.getMinReduction(), 0.01);
      assertEquals(10.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      SecondOrderDifferentialEquations secondOrderDifferentialEquations0 = mock(SecondOrderDifferentialEquations.class, new ViolatedAssumptionAnswer());
      doReturn(1).when(secondOrderDifferentialEquations0).getDimension();
      FirstOrderConverter firstOrderConverter0 = new FirstOrderConverter(secondOrderDifferentialEquations0);
      DormandPrince853Integrator dormandPrince853Integrator0 = new DormandPrince853Integrator((-32.81567793066633), (-32.81567793066633), (-32.81567793066633), 1);
      double[][] doubleArray0 = new double[6][2];
      double double0 = dormandPrince853Integrator0.integrate(firstOrderConverter0, 1546.0, doubleArray0[4], 1, doubleArray0[0]);
      assertEquals(590, dormandPrince853Integrator0.getEvaluations());
      assertEquals(0.9999999999999996, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      SecondOrderDifferentialEquations secondOrderDifferentialEquations0 = mock(SecondOrderDifferentialEquations.class, new ViolatedAssumptionAnswer());
      doReturn(1).when(secondOrderDifferentialEquations0).getDimension();
      FirstOrderConverter firstOrderConverter0 = new FirstOrderConverter(secondOrderDifferentialEquations0);
      DormandPrince853Integrator dormandPrince853Integrator0 = new DormandPrince853Integrator(0.029394451945712108, 0.029394451945712108, 0.029394451945712108, 1);
      FixedStepHandler fixedStepHandler0 = mock(FixedStepHandler.class, new ViolatedAssumptionAnswer());
      StepNormalizer stepNormalizer0 = new StepNormalizer(0.029394451945712108, fixedStepHandler0);
      dormandPrince853Integrator0.addStepHandler(stepNormalizer0);
      double[][] doubleArray0 = new double[6][2];
      double double0 = dormandPrince853Integrator0.integrate(firstOrderConverter0, 0.029394451945712108, doubleArray0[4], 1, doubleArray0[0]);
      assertEquals(626, dormandPrince853Integrator0.getEvaluations());
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      SecondOrderDifferentialEquations secondOrderDifferentialEquations0 = mock(SecondOrderDifferentialEquations.class, new ViolatedAssumptionAnswer());
      doReturn(1).when(secondOrderDifferentialEquations0).getDimension();
      FirstOrderConverter firstOrderConverter0 = new FirstOrderConverter(secondOrderDifferentialEquations0);
      DormandPrince853Integrator dormandPrince853Integrator0 = new DormandPrince853Integrator(0.029394451945712108, 0.029394451945712108, 0.029394451945712108, 1);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(0.0, 0.0, 0.0, 0.0, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      dormandPrince853Integrator0.addEventHandler(eventHandler0, 1, 0.029394451945712108, 1);
      double[][] doubleArray0 = new double[6][2];
      double double0 = dormandPrince853Integrator0.integrate(firstOrderConverter0, 0.029394451945712108, doubleArray0[4], 1, doubleArray0[0]);
      assertEquals(527, dormandPrince853Integrator0.getEvaluations());
      assertEquals(1.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      SecondOrderDifferentialEquations secondOrderDifferentialEquations0 = mock(SecondOrderDifferentialEquations.class, new ViolatedAssumptionAnswer());
      doReturn(1).when(secondOrderDifferentialEquations0).getDimension();
      FirstOrderConverter firstOrderConverter0 = new FirstOrderConverter(secondOrderDifferentialEquations0);
      DormandPrince853Integrator dormandPrince853Integrator0 = new DormandPrince853Integrator(0.029394451945712108, 1718.117905, 1718.117905, 1);
      double[] doubleArray0 = new double[2];
      doubleArray0[0] = 1718.117905;
      doubleArray0[1] = 0.029394451945712108;
      double double0 = dormandPrince853Integrator0.integrate(firstOrderConverter0, 0.029394451945712108, doubleArray0, 1819.1233093573749, doubleArray0);
      assertEquals(206, dormandPrince853Integrator0.getEvaluations());
      assertEquals(3537.2412143573706, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      SecondOrderDifferentialEquations secondOrderDifferentialEquations0 = mock(SecondOrderDifferentialEquations.class, new ViolatedAssumptionAnswer());
      doReturn(1).when(secondOrderDifferentialEquations0).getDimension();
      FirstOrderConverter firstOrderConverter0 = new FirstOrderConverter(secondOrderDifferentialEquations0);
      double[] doubleArray0 = new double[2];
      HighamHall54Integrator highamHall54Integrator0 = new HighamHall54Integrator(737.446043, doubleArray0[1], 1546.0, (-166.91112150209));
      try { 
        highamHall54Integrator0.integrate(firstOrderConverter0, 1544.0893606235243, doubleArray0, 737.446043, doubleArray0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // minimal step size (7.37E02) reached, integration needs 0.00E00
         //
         verifyException("org.apache.commons.math.ode.nonstiff.AdaptiveStepsizeIntegrator", e);
      }
  }
}
