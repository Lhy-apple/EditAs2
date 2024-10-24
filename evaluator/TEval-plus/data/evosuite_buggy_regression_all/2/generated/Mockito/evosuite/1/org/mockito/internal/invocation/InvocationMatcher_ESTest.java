/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:34:52 GMT 2023
 */

package org.mockito.internal.invocation;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Vector;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.hamcrest.object.IsEventFrom;
import org.junit.runner.RunWith;
import org.mockito.internal.creation.util.MockitoMethodProxy;
import org.mockito.internal.invocation.InvocationImpl;
import org.mockito.internal.invocation.InvocationMatcher;
import org.mockito.internal.invocation.MockitoMethod;
import org.mockito.internal.invocation.realmethod.CleanTraceRealMethod;
import org.mockito.internal.invocation.realmethod.DefaultRealMethod;
import org.mockito.internal.stubbing.StubbedInvocationMatcher;
import org.mockito.invocation.Invocation;
import org.mockito.invocation.Location;
import org.mockito.stubbing.Answer;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class InvocationMatcher_ESTest extends InvocationMatcher_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
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
  public void test1()  throws Throwable  {
      Object[] objectArray0 = new Object[0];
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(mockitoMethod0).isVarArgs();
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      DefaultRealMethod defaultRealMethod0 = new DefaultRealMethod(mockitoMethodProxy0);
      CleanTraceRealMethod cleanTraceRealMethod0 = new CleanTraceRealMethod(defaultRealMethod0);
      InvocationImpl invocationImpl0 = new InvocationImpl((Object) null, mockitoMethod0, objectArray0, 1, cleanTraceRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      Location location0 = invocationMatcher0.getLocation();
      assertNotNull(location0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Object[] objectArray0 = new Object[1];
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(mockitoMethod0).isVarArgs();
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      DefaultRealMethod defaultRealMethod0 = new DefaultRealMethod(mockitoMethodProxy0);
      CleanTraceRealMethod cleanTraceRealMethod0 = new CleanTraceRealMethod(defaultRealMethod0);
      InvocationImpl invocationImpl0 = new InvocationImpl(cleanTraceRealMethod0, mockitoMethod0, objectArray0, 5, defaultRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      // Undeclared exception!
      try { 
        invocationMatcher0.toString();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Object[] objectArray0 = new Object[1];
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      doReturn((Object) null).when(invocation0).getMock();
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn(true).when(mockitoMethod0).isVarArgs();
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      DefaultRealMethod defaultRealMethod0 = new DefaultRealMethod(mockitoMethodProxy0);
      InvocationImpl invocationImpl0 = new InvocationImpl(mockitoMethod0, mockitoMethod0, objectArray0, 21, defaultRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      Answer<IsEventFrom> answer0 = (Answer<IsEventFrom>) mock(Answer.class, new ViolatedAssumptionAnswer());
      StubbedInvocationMatcher stubbedInvocationMatcher0 = new StubbedInvocationMatcher(invocationMatcher0, answer0);
      boolean boolean0 = stubbedInvocationMatcher0.matches(invocation0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Object[] objectArray0 = new Object[0];
      MockitoMethod mockitoMethod0 = mock(MockitoMethod.class, new ViolatedAssumptionAnswer());
      doReturn((Method) null, (Method) null).when(mockitoMethod0).getJavaMethod();
      doReturn(true).when(mockitoMethod0).isVarArgs();
      MockitoMethodProxy mockitoMethodProxy0 = mock(MockitoMethodProxy.class, new ViolatedAssumptionAnswer());
      CleanTraceRealMethod cleanTraceRealMethod0 = new CleanTraceRealMethod(mockitoMethodProxy0);
      InvocationImpl invocationImpl0 = new InvocationImpl(mockitoMethod0, mockitoMethod0, objectArray0, (-5), cleanTraceRealMethod0);
      InvocationMatcher invocationMatcher0 = new InvocationMatcher(invocationImpl0);
      // Undeclared exception!
      try { 
        invocationMatcher0.matches(invocationImpl0);
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
      Vector<Invocation> vector0 = new Vector<Invocation>();
      List<InvocationMatcher> list0 = InvocationMatcher.createFrom(vector0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      Vector<Invocation> vector0 = new Vector<Invocation>();
      vector0.add((Invocation) null);
      // Undeclared exception!
      try { 
        InvocationMatcher.createFrom(vector0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }
}
