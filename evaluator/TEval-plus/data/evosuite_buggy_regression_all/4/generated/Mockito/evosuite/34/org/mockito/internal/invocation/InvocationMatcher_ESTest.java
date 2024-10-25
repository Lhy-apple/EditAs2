/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:19:09 GMT 2023
 */

package org.mockito.internal.invocation;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Method;
import java.math.RoundingMode;
import java.util.LinkedHashSet;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.hamcrest.object.IsEventFrom;
import org.hamcrest.xml.HasXPath;
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>(404, 404);
      Class<HasXPath> class0 = HasXPath.class;
      IsEventFrom isEventFrom0 = new IsEventFrom(class0, linkedHashSet0);
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((Method) null).when(invocation0).getMethod();
      doReturn(isEventFrom0).when(invocation0).getMock();
      Invocation invocation1 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation1).argumentsToMatchers();
      doReturn((Method) null).when(invocation1).getMethod();
      doReturn(isEventFrom0).when(invocation1).getMock();
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
      Answer<RoundingMode> answer0 = (Answer<RoundingMode>) mock(Answer.class, new ViolatedAssumptionAnswer());
      StubbedInvocationMatcher stubbedInvocationMatcher0 = null;
      try {
        stubbedInvocationMatcher0 = new StubbedInvocationMatcher(invocationMatcher0, answer0);
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
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Class<HasXPath> class0 = HasXPath.class;
      IsEventFrom isEventFrom0 = new IsEventFrom(class0, linkedHashSet0);
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(invocation0).getMock();
      Invocation invocation1 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation1).argumentsToMatchers();
      doReturn(isEventFrom0).when(invocation1).getMock();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation1);
      boolean boolean0 = invocationMatcher0.matches(invocation0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Class<HasXPath> class0 = HasXPath.class;
      IsEventFrom isEventFrom0 = new IsEventFrom(class0, linkedHashSet0);
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      Invocation invocation1 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((List) null).when(invocation1).argumentsToMatchers();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation1);
      invocationMatcher0.captureArgumentsFrom(invocation0);
  }
}
