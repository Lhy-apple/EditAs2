/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:14:20 GMT 2023
 */

package org.mockito.internal.invocation;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.EventObject;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.hamcrest.object.IsEventFrom;
import org.hamcrest.text.IsEqualIgnoringCase;
import org.junit.runner.RunWith;
import org.mockito.internal.debugging.Location;
import org.mockito.internal.invocation.Invocation;
import org.mockito.internal.invocation.InvocationMatcher;
import org.mockito.internal.reporting.PrintSettings;
import org.mockito.internal.stubbing.StubbedInvocationMatcher;
import org.mockito.stubbing.Answer;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InvocationMatcher_ESTest extends InvocationMatcher_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation0).argumentsToMatchers();
      doReturn((Method) null).when(invocation0).getMethod();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      // Undeclared exception!
      try { 
        invocationMatcher0.hasSimilarMethod(invocation0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      IsEqualIgnoringCase isEqualIgnoringCase0 = new IsEqualIgnoringCase("");
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((Method) null).when(invocation0).getMethod();
      doReturn(isEqualIgnoringCase0).when(invocation0).getMock();
      Invocation invocation1 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation1).argumentsToMatchers();
      doReturn((Method) null).when(invocation1).getMethod();
      doReturn(isEqualIgnoringCase0).when(invocation1).getMock();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation1);
      // Undeclared exception!
      try { 
        invocationMatcher0.matches(invocation0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation0).argumentsToMatchers();
      doReturn((String) null).when(invocation0).toString(anyList() , any(org.mockito.internal.reporting.PrintSettings.class));
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      PrintSettings printSettings0 = new PrintSettings();
      String string0 = invocationMatcher0.toString(printSettings0);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation0).argumentsToMatchers();
      doReturn((String) null).when(invocation0).toString();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      StubbedInvocationMatcher stubbedInvocationMatcher0 = null;
      try {
        stubbedInvocationMatcher0 = new StubbedInvocationMatcher(invocationMatcher0, (Answer) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation0).argumentsToMatchers();
      doReturn((String) null).when(invocation0).toString(anyList() , any(org.mockito.internal.reporting.PrintSettings.class));
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      String string0 = invocationMatcher0.toString();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation0).argumentsToMatchers();
      doReturn((Location) null).when(invocation0).getLocation();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      Location location0 = invocationMatcher0.getLocation();
      assertNull(location0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      IsEventFrom isEventFrom0 = new IsEventFrom(class0, class0);
      EventObject eventObject0 = new EventObject(isEventFrom0);
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation0).argumentsToMatchers();
      doReturn(eventObject0, class0).when(invocation0).getMock();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      boolean boolean0 = invocationMatcher0.matches(invocation0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation0).argumentsToMatchers();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      invocationMatcher0.captureArgumentsFrom((Invocation) null);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      LinkedList<Invocation> linkedList0 = new LinkedList<Invocation>();
      List<InvocationMatcher> list0 = InvocationMatcher.createFrom(linkedList0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      ArrayList<Invocation> arrayList0 = new ArrayList<Invocation>();
      arrayList0.add((Invocation) null);
      // Undeclared exception!
      try { 
        InvocationMatcher.createFrom(arrayList0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }
}
