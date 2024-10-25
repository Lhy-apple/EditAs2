/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:53:18 GMT 2023
 */

package org.mockito.internal.invocation;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Stack;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.hamcrest.beans.SamePropertyValuesAs;
import org.hamcrest.core.StringStartsWith;
import org.junit.runner.RunWith;
import org.mockito.internal.creation.util.MockitoMethodProxy;
import org.mockito.internal.invocation.InvocationImpl;
import org.mockito.internal.invocation.InvocationMatcher;
import org.mockito.internal.invocation.MockitoMethod;
import org.mockito.internal.invocation.realmethod.DefaultRealMethod;
import org.mockito.internal.stubbing.StubbedInvocationMatcher;
import org.mockito.invocation.Invocation;
import org.mockito.stubbing.Answer;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InvocationMatcher_ESTest extends InvocationMatcher_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      DefaultRealMethod defaultRealMethod0 = new DefaultRealMethod(mockitoMethodProxy0);
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn((Method) null).when(mockitoMethod0).getJavaMethod();
      doReturn(false).when(mockitoMethod0).isVarArgs();
      InvocationImpl invocationImpl0 = new InvocationImpl(defaultRealMethod0, mockitoMethod0, (Object[]) null, 277, defaultRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      SamePropertyValuesAs<Object> samePropertyValuesAs0 = new SamePropertyValuesAs<Object>(invocationMatcher0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      DefaultRealMethod defaultRealMethod0 = new DefaultRealMethod(mockitoMethodProxy0);
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(mockitoMethod0).isVarArgs();
      InvocationImpl invocationImpl0 = new InvocationImpl(mockitoMethod0, mockitoMethod0, (Object[]) null, 305, defaultRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      // Undeclared exception!
      try { 
        invocationMatcher0.toString();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      DefaultRealMethod defaultRealMethod0 = new DefaultRealMethod(mockitoMethodProxy0);
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(invocation0).getMock();
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn(false).when(mockitoMethod0).isVarArgs();
      InvocationImpl invocationImpl0 = new InvocationImpl(mockitoMethod0, mockitoMethod0, (Object[]) null, 2801, defaultRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      boolean boolean0 = invocationMatcher0.matches(invocation0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      DefaultRealMethod defaultRealMethod0 = new DefaultRealMethod(mockitoMethodProxy0);
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn((Method) null, (Method) null).when(mockitoMethod0).getJavaMethod();
      doReturn(false).when(mockitoMethod0).isVarArgs();
      Object[] objectArray0 = new Object[8];
      InvocationImpl invocationImpl0 = new InvocationImpl(defaultRealMethod0, mockitoMethod0, objectArray0, 1, defaultRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      Answer<StringStartsWith> answer0 = (Answer<StringStartsWith>) mock(Answer.class, new ViolatedAssumptionAnswer());
      StubbedInvocationMatcher stubbedInvocationMatcher0 = new StubbedInvocationMatcher(invocationMatcher0, answer0);
      // Undeclared exception!
      try { 
        stubbedInvocationMatcher0.matches(invocationImpl0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Stack<Invocation> stack0 = new Stack<Invocation>();
      List<InvocationMatcher> list0 = InvocationMatcher.createFrom(stack0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Stack<Invocation> stack0 = new Stack<Invocation>();
      stack0.add((Invocation) null);
      // Undeclared exception!
      try { 
        InvocationMatcher.createFrom(stack0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }
}
