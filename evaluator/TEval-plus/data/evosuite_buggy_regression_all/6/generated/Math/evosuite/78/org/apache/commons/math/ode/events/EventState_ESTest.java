/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:54:50 GMT 2023
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
      EventState eventState0 = new EventState(eventHandler0, 858.254009893, 82.0, 128);
      eventState0.getEventHandler();
      assertEquals(82.0, eventState0.getConvergence(), 0.01);
      assertFalse(eventState0.stop());
      assertEquals(858.254009893, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(128, eventState0.getMaxIterationCount());
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, (-621), (-621), (-621));
      int int0 = eventState0.getMaxIterationCount();
      assertFalse(eventState0.stop());
      assertEquals((-621), int0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals((-621.0), eventState0.getMaxCheckInterval(), 0.01);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 456.8849, 456.8849, 645);
      double double0 = eventState0.getEventTime();
      assertEquals(645, eventState0.getMaxIterationCount());
      assertEquals(456.8849, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(Double.NaN, double0, 0.01);
      assertFalse(eventState0.stop());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 3216);
      double double0 = eventState0.getMaxCheckInterval();
      assertFalse(eventState0.stop());
      assertEquals(Double.POSITIVE_INFINITY, double0, 0.01);
      assertEquals(3216, eventState0.getMaxIterationCount());
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 858.254009893, 82.0, 128);
      double double0 = eventState0.getConvergence();
      assertEquals(128, eventState0.getMaxIterationCount());
      assertFalse(eventState0.stop());
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(858.254009893, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(82.0, double0, 0.01);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      double[] doubleArray0 = new double[22];
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn((double)(-1073741882)).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, (-1073741882), 1.0, 1211);
      eventState0.reinitializeBegin(1.0, doubleArray0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals((-1.073741882E9), eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(1211, eventState0.getMaxIterationCount());
      assertFalse(eventState0.stop());
      assertEquals(1.0, eventState0.getConvergence(), 0.01);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      double[] doubleArray0 = new double[5];
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 3606.349678368603, 3606.349678368603, 4756);
      eventState0.reinitializeBegin(3606.349678368603, doubleArray0);
      assertEquals(4756, eventState0.getMaxIterationCount());
      assertEquals(3606.349678368603, eventState0.getMaxCheckInterval(), 0.01);
      assertFalse(eventState0.stop());
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(3605.86764637416).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 3605.86764637416, 3605.86764637416, 4756);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      
      double[] doubleArray0 = new double[5];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals(4756, eventState0.getMaxIterationCount());
      assertFalse(eventState0.stop());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn((-1.0), 0.0, (-1762.0835), (-1762.0835)).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, 0);
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      eventState0.stepAccepted((-1.0), doubleArray0);
      // Undeclared exception!
      try { 
        eventState0.evaluateStep(dummyStepInterpolator0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // function values at endpoints do not have different signs.  Endpoints: [\uFFFD, -1], Values: [-1,762.083, -1,762.083]
         //
         verifyException("org.apache.commons.math.MathRuntimeException", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(3605.86764637416, (-1.0), (-1.0), 3605.86764637416, 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 3605.86764637416, 3605.86764637416, 4756);
      double[] doubleArray0 = new double[5];
      eventState0.stepAccepted((-1.0), doubleArray0);
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      dummyStepInterpolator0.storeTime(3605.86764637416);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals(1801.93382318708, dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn((-1.0), 0.0, (-1762.0835)).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, 0);
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals(Double.NaN, dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      
      boolean boolean1 = eventState0.reset(0.0, doubleArray0);
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
      assertFalse(boolean1 == boolean0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertFalse(boolean1);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(2).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-1.0), 0.0, (-1762.0835), (-1762.0835), 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, 0);
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      eventState0.evaluateStep(dummyStepInterpolator0);
      eventState0.stepAccepted((-1.0), doubleArray0);
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals((-1.0), dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(2).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-1.0), 0.0, (-1762.0835), (-1762.0835), 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, 2);
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      eventState0.evaluateStep(dummyStepInterpolator0);
      eventState0.stepAccepted((-1.0), doubleArray0);
      dummyStepInterpolator0.storeTime((-2580.38885814));
      boolean boolean0 = eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals((-1.0), dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-1.0), 0.0, (-1.0), (-1.0)).when(eventHandler0).g(anyDouble() , any(double[].class));
      double[] doubleArray0 = new double[10];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, false);
      EventState eventState0 = new EventState(eventHandler0, 0.0, (-1), (-1));
      eventState0.evaluateStep(dummyStepInterpolator0);
      assertFalse(eventState0.stop());
      
      eventState0.stepAccepted(1802.2228228223134, doubleArray0);
      assertTrue(eventState0.stop());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, 0);
      boolean boolean0 = eventState0.stop();
      assertFalse(boolean0);
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(0, eventState0.getMaxIterationCount());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(0).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-1.0), 0.0, (-1762.0835), 0.0).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, 0);
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      eventState0.evaluateStep(dummyStepInterpolator0);
      assertFalse(eventState0.stop());
      
      eventState0.stepAccepted(1.0, doubleArray0);
      boolean boolean0 = eventState0.stop();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      EventState eventState0 = new EventState((EventHandler) null, (-1.0), (-253.96117164), 4756);
      double[] doubleArray0 = new double[1];
      boolean boolean0 = eventState0.reset(5409.024517552904, doubleArray0);
      assertFalse(boolean0);
      assertEquals(4756, eventState0.getMaxIterationCount());
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals((-1.0), eventState0.getMaxCheckInterval(), 0.01);
      assertFalse(eventState0.stop());
      assertEquals(253.96117164, eventState0.getConvergence(), 0.01);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(1).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((double)(-1073741824), 0.0, 1.0E-6, 2198.089263205512).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, (-5530.87841266201), (-1073741824));
      double[] doubleArray0 = new double[0];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      eventState0.evaluateStep(dummyStepInterpolator0);
      assertEquals(Double.NaN, dummyStepInterpolator0.getInterpolatedTime(), 0.01);
      
      eventState0.stepAccepted((-3797.3), doubleArray0);
      boolean boolean0 = eventState0.reset((-5530.87841266201), doubleArray0);
      assertTrue(boolean0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
      assertEquals(5530.87841266201, eventState0.getConvergence(), 0.01);
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
      assertEquals((-1073741824), eventState0.getMaxIterationCount());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      EventHandler eventHandler0 = mock(EventHandler.class, new ViolatedAssumptionAnswer());
      doReturn(2).when(eventHandler0).eventOccurred(anyDouble() , any(double[].class) , anyBoolean());
      doReturn((-1.0), 0.0, (-1762.0835), (-1762.0835)).when(eventHandler0).g(anyDouble() , any(double[].class));
      EventState eventState0 = new EventState(eventHandler0, 0.0, 0.0, 0);
      double[] doubleArray0 = new double[9];
      DummyStepInterpolator dummyStepInterpolator0 = new DummyStepInterpolator(doubleArray0, true);
      eventState0.evaluateStep(dummyStepInterpolator0);
      eventState0.stepAccepted((-1.0), doubleArray0);
      boolean boolean0 = eventState0.reset(0.0, doubleArray0);
      assertEquals(0, eventState0.getMaxIterationCount());
      assertEquals(0.0, eventState0.getMaxCheckInterval(), 0.01);
      assertTrue(boolean0);
      assertEquals(Double.NaN, eventState0.getEventTime(), 0.01);
  }
}
