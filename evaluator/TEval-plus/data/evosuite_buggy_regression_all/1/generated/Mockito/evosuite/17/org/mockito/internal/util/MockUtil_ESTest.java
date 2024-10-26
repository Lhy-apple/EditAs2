/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:12:35 GMT 2023
 */

package org.mockito.internal.util;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.mockito.internal.creation.MethodInterceptorFilter;
import org.mockito.internal.creation.MockSettingsImpl;
import org.mockito.internal.util.MockUtil;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MockUtil_ESTest extends MockUtil_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      MockUtil mockUtil0 = new MockUtil();
      // Undeclared exception!
      try { 
        mockUtil0.resetMock("gbiD8");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      MockUtil mockUtil0 = new MockUtil();
      Class<MethodInterceptorFilter> class0 = MethodInterceptorFilter.class;
      MockSettingsImpl mockSettingsImpl0 = new MockSettingsImpl();
      mockSettingsImpl0.serializable();
      // Undeclared exception!
      try { 
        mockUtil0.createMock(class0, mockSettingsImpl0);
        fail("Expecting exception: IncompatibleClassChangeError");
      
      } catch(IncompatibleClassChangeError e) {
         //
         // Expected non-static field org.mockito.cglib.proxy.Enhancer.serialVersionUID
         //
         verifyException("org.mockito.cglib.proxy.Enhancer", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      MockUtil mockUtil0 = new MockUtil();
      Class<Object> class0 = Object.class;
      MockSettingsImpl mockSettingsImpl0 = new MockSettingsImpl();
      // Undeclared exception!
      try { 
        mockUtil0.createMock(class0, mockSettingsImpl0);
        fail("Expecting exception: IncompatibleClassChangeError");
      
      } catch(IncompatibleClassChangeError e) {
         //
         // Expected non-static field org.mockito.cglib.proxy.Enhancer.serialVersionUID
         //
         verifyException("org.mockito.cglib.proxy.Enhancer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      MockUtil mockUtil0 = new MockUtil();
      // Undeclared exception!
      try { 
        mockUtil0.getMockName((Object) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      MockUtil mockUtil0 = new MockUtil();
      boolean boolean0 = mockUtil0.isMock((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      MockUtil mockUtil0 = new MockUtil();
      boolean boolean0 = mockUtil0.isMock(mockUtil0);
      assertFalse(boolean0);
  }
}
