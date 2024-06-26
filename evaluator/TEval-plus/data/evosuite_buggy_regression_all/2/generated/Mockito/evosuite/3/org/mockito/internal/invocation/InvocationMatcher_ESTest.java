/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:34:54 GMT 2023
 */

package org.mockito.internal.invocation;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.hamcrest.core.StringStartsWith;
import org.hamcrest.number.BigDecimalCloseTo;
import org.hamcrest.text.IsEmptyString;
import org.junit.runner.RunWith;
import org.mockito.internal.invocation.InvocationMatcher;
import org.mockito.internal.stubbing.StubbedInvocationMatcher;
import org.mockito.invocation.Invocation;
import org.mockito.invocation.Location;
import org.mockito.stubbing.Answer;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InvocationMatcher_ESTest extends InvocationMatcher_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Object[] objectArray0 = new Object[2];
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      Invocation invocation1 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn(objectArray0).when(invocation1).getArguments();
      doReturn((Method) null).when(invocation1).getMethod();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation1);
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
      Object[] objectArray0 = new Object[2];
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((String) null).when(invocation0).toString();
      doReturn(objectArray0).when(invocation0).getArguments();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      Answer<IsEmptyString> answer0 = (Answer<IsEmptyString>) mock(Answer.class, new ViolatedAssumptionAnswer());
      StubbedInvocationMatcher stubbedInvocationMatcher0 = new StubbedInvocationMatcher(invocationMatcher0, answer0);
      assertFalse(stubbedInvocationMatcher0.wasUsed());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Object[] objectArray0 = new Object[1];
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((Object[]) null).when(invocation0).getArguments();
      InvocationMatcher invocationMatcher0 = null;
      try {
        invocationMatcher0 = new InvocationMatcher(invocation0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.ArgumentsProcessor", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Object[] objectArray0 = new Object[2];
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn(objectArray0).when(invocation0).getArguments();
      doReturn((Object) null).when(invocation0).getMock();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation0);
      // Undeclared exception!
      try { 
        invocationMatcher0.toString();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      StringStartsWith stringStartsWith0 = new StringStartsWith("");
      Object[] objectArray0 = new Object[3];
      objectArray0[1] = (Object) stringStartsWith0;
      BigDecimalCloseTo bigDecimalCloseTo0 = new BigDecimalCloseTo((BigDecimal) null, (BigDecimal) null);
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn(bigDecimalCloseTo0).when(invocation0).getMock();
      RoundingMode roundingMode0 = RoundingMode.UNNECESSARY;
      Invocation invocation1 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn(objectArray0).when(invocation1).getArguments();
      doReturn((Method) null, (Method) null).when(invocation1).getMethod();
      doReturn(roundingMode0, stringStartsWith0, objectArray0[1]).when(invocation1).getMock();
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocation1);
      invocationMatcher0.matches(invocation0);
      // Undeclared exception!
      try { 
        invocationMatcher0.matches(invocation1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      LinkedList<Invocation> linkedList0 = new LinkedList<Invocation>();
      List<InvocationMatcher> list0 = InvocationMatcher.createFrom(linkedList0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      LinkedList<Invocation> linkedList0 = new LinkedList<Invocation>();
      linkedList0.add((Invocation) null);
      // Undeclared exception!
      try { 
        InvocationMatcher.createFrom(linkedList0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }
}
