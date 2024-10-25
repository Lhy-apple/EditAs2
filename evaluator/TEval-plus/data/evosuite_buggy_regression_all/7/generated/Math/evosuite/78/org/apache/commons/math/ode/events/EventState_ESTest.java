/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 20:00:25 GMT 2023
 */

package org.apache.commons.math.ode.events;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math.ode.events.EventHandler;
import org.apache.commons.math.ode.events.EventState;
import org.apache.commons.math.ode.sampling.DummyStepInterpolator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class EventState_ESTest extends EventState_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(eventHandler0).toString();
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, (-2596));
      eventState0.getEventHandler();
      assertEquals((-2596), eventState0.getMaxIterationCount());
      assertFalse(eventState0.stop());
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 1288490189, 1288490189, 1288490189);
      int int0 = eventState0.getMaxIterationCount();
      assertEquals(1.288490189E9, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertFalse(eventState0.stop());
      assertEquals(1288490189, int0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      EventState eventState0 = new EventState((EventHandler) null, (-3130.486), (-3130.486), (-1));
      double double0 = eventState0.getEventTime();
      assertEquals((-1), eventState0.getMaxIterationCount());
      assertFalse(eventState0.stop());
      assertEquals(Double.NaN, double0, 0.01);
      assertEquals((-3130.486), eventState0.getMaxCheckInterval(), 0.01);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 1298.2583436762325, 1298.2583436762325, 2739);
      double double0 = eventState0.getMaxCheckInterval();
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(2739, eventState0.getMaxIterationCount());
      assertEquals(1298.2583436762325, double0, 0.01);
      assertFalse(eventState0.stop());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 0.0, 1202.0, 0);
      double double0 = eventState0.getConvergence();
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertFalse(eventState0.stop());
      assertEquals(0, eventState0.getMaxIterationCount());
      assertEquals(1202.0, double0, 0.01);
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn((-557.47260976)).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, (-1.0), (-1.0), 2146155616);
      eventState0.reinitializeBegin(2146155616, (double[]) null);
      assertFalse(eventState0.stop());
      assertEquals(2146155616, eventState0.getMaxIterationCount());
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals((-1.0), eventState0.getMaxCheckInterval(), 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 1298.2583436762325, 1298.2583436762325, 2739);
      double[] doubleArray0 = new double[6];
      eventState0.reinitializeBegin(Double.NaN, doubleArray0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertFalse(eventState0.stop());
      assertEquals(1298.2583436762325, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(2739, eventState0.getMaxIterationCount());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn((-2141.8946422), (-2141.8946422), (double)1105, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, (-2141.8946422), 1316.8463, 1105);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      boolean boolean1 = eventState0.reset(0.0, doubleArray0);
      assertEquals(1316.8463, eventState0.getConvergence(), 0.01);
      assertEquals(1105, eventState0.getMaxIterationCount());
      assertEquals((-2141.8946422), eventState0.getMaxCheckInterval(), 0.01);
      assertFalse(boolean1 == boolean0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(1105).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-2141.8946422), (-2141.8946422), (double)1105, 0.0, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, (-2141.8946422), 1316.8463, 1105);
      eventState0.evaluateStep(dummyStepInterpolator0);
      eventState0.stepAccepted(0.0, doubleArray0);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals(0.0, dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      dummyStepInterpolator0.storeTime(1105);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(1105).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-2141.8946422), (-2141.8946422), (double)1105, 0.0, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, (-2141.8946422), 1316.8463, 1105);
      eventState0.evaluateStep(dummyStepInterpolator0);
      eventState0.stepAccepted(0.0, doubleArray0);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals(2421.8463, dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(2.2250738585072014E-308, 2.2250738585072014E-308, (-1536.1491012569882), 1909.107237959299, 2.2250738585072014E-308).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, (-2596));
      double[] doubleArray0 = new double[4];
      eventState0.stepAccepted(1.0, doubleArray0);
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, false);
      eventState0.evaluateStep(dummyStepInterpolator0);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals(1.0, dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, false);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(1105).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-2141.8946422), (-2141.8946422), (double)1105, 0.0, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      dummyStepInterpolator0.storeTime((-1.0));
      EventState eventState0 = new EventState(eventHandler0, (-2141.8946422), 1316.8463, 1105);
      eventState0.evaluateStep(dummyStepInterpolator0);
      eventState0.stepAccepted(1105, doubleArray0);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals((-1317.8463), eventState0.getEventTime(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, false);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn((-2141.8946422), 0.0, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      dummyStepInterpolator0.storeTime((-2141.8946422));
      EventState eventState0 = new EventState(eventHandler0, (-2141.8946422), 1316.8463, 1105);
      // Undeclared exception!
      try { 
        eventState0.evaluateStep(dummyStepInterpolator0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // function values at endpoints do not have different signs.  Endpoints: [\uFFFD, \uFFFD], Values: [-2,141.895, -2,141.895]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(2.2250738585072014E-308, 2.2250738585072014E-308, (-1536.1491012569882)).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, (-2596));
      double[] doubleArray0 = new double[4];
      eventState0.stepAccepted(0.0, doubleArray0);
      eventState0.stepAccepted(1532.0474470121567, doubleArray0);
      eventState0.stepAccepted(1.0, doubleArray0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals((-2596), eventState0.getMaxIterationCount());
      assertFalse(eventState0.stop());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, (-2596));
      boolean boolean0 = eventState0.stop();
      assertEquals((-2596), eventState0.getMaxIterationCount());
      assertFalse(boolean0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn(2.2250738585072014E-308, 2.2250738585072014E-308, (-1536.1491012569882), 1909.107237959299, 2.2250738585072014E-308).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, (-2596));
      double[] doubleArray0 = new double[4];
      eventState0.stepAccepted(0.0, doubleArray0);
      eventState0.stepAccepted(1532.0474470121567, doubleArray0);
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, false);
      eventState0.evaluateStep(dummyStepInterpolator0);
      eventState0.stepAccepted(2.2250738585072014E-308, doubleArray0);
      boolean boolean0 = eventState0.stop();
      assertEquals(1532.0474470121567, eventState0.getEventTime(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, (-2596));
      double[] doubleArray0 = new double[4];
      boolean boolean0 = eventState0.reset(1.0508919439394515, doubleArray0);
      assertFalse(eventState0.stop());
      assertEquals((-2596), eventState0.getMaxIterationCount());
      assertFalse(boolean0);
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, false);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(1).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-2141.8946422), (-2141.8946422), (double)1105, 0.0, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, (-2141.8946422), 1316.8463, 1105);
      eventState0.evaluateStep(dummyStepInterpolator0);
      assertFalse(eventState0.stop());
      
      eventState0.stepAccepted((-2141.8946422), doubleArray0);
      boolean boolean0 = eventState0.reset((-3274.0), doubleArray0);
      assertTrue(boolean0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(1105, eventState0.getMaxIterationCount());
      assertEquals((-2141.8946422), eventState0.getMaxCheckInterval(), 0.01);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, false);
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(2).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-2141.8946422), (-2141.8946422), (double)1105, 0.0, (-2141.8946422)).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, (-2141.8946422), 1316.8463, 1105);
      eventState0.evaluateStep(dummyStepInterpolator0);
      assertFalse(eventState0.stop());
      
      eventState0.stepAccepted((-2141.8946422), doubleArray0);
      boolean boolean0 = eventState0.reset((-3274.0), doubleArray0);
      assertTrue(boolean0);
      assertEquals(1105, eventState0.getMaxIterationCount());
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(1316.8463, eventState0.getConvergence(), 0.01);
  }
}
