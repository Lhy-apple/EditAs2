/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:36:43 GMT 2023
 */

package org.mockito.internal;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.shaded.org.mockito.Mockito.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.annotation.Annotation;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.ViolatedAssumptionAnswer;
import org.hamcrest.Matcher;
import org.junit.runner.RunWith;
import org.mockito.internal.MockHandler;
import org.mockito.internal.creation.MockSettingsImpl;
import org.mockito.internal.invocation.Invocation;
import org.mockito.internal.stubbing.InvocationContainerImpl;
import org.mockito.stubbing.Answer;
import org.mockito.stubbing.VoidMethodStubbable;
import org.mockito.stubbing.answers.ReturnsElementsOf;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MockHandler_ESTest extends MockHandler_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MockHandler<Object> mockHandler0 = new MockHandler<Object>();
      Locale.FilteringMode locale_FilteringMode0 = Locale.FilteringMode.AUTOSELECT_FILTERING;
      VoidMethodStubbable<Object> voidMethodStubbable0 = mockHandler0.voidMethodStubbable(locale_FilteringMode0);
      assertNotNull(voidMethodStubbable0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      MockHandler<String> mockHandler0 = new MockHandler<String>();
      MockHandler<String> mockHandler1 = new MockHandler<String>(mockHandler0);
      assertFalse(mockHandler1.equals((Object)mockHandler0));
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      MockHandler<Matcher<Object>> mockHandler0 = new MockHandler<Matcher<Object>>();
      // Undeclared exception!
      try { 
        mockHandler0.setAnswersForStubbing((List<Answer>) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      MockHandler<Integer> mockHandler0 = new MockHandler<Integer>((MockSettingsImpl) null);
      InvocationContainerImpl invocationContainerImpl0 = (InvocationContainerImpl)mockHandler0.getInvocationContainer();
      assertFalse(invocationContainerImpl0.hasAnswersForStubbing());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Invocation invocation0 = mock(Invocation.class, new ViolatedAssumptionAnswer());
      MockHandler<Object> mockHandler0 = new MockHandler<Object>();
      try { 
        mockHandler0.handle(invocation0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.MockHandler", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      MockHandler<Object> mockHandler0 = new MockHandler<Object>();
      MockHandler<String> mockHandler1 = new MockHandler<String>();
      InvocationContainerImpl invocationContainerImpl0 = mockHandler1.invocationContainerImpl;
      ArrayList<Annotation> arrayList0 = new ArrayList<Annotation>();
      ReturnsElementsOf returnsElementsOf0 = new ReturnsElementsOf(arrayList0);
      invocationContainerImpl0.addAnswerForVoidMethod(returnsElementsOf0);
      mockHandler0.invocationContainerImpl = invocationContainerImpl0;
      try { 
        mockHandler0.handle((Invocation) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.invocation.InvocationMatcher", e);
      }
  }
}