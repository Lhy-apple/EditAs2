/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:54:22 GMT 2023
 */

package org.mockito.internal;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.mockito.MockSettings;
import org.mockito.internal.MockitoCore;
import org.mockito.internal.creation.MockSettingsImpl;
import org.mockito.internal.invocation.Invocation;
import org.mockito.internal.verification.Only;
import org.mockito.internal.verification.api.VerificationMode;
import org.mockito.stubbing.Answer;
import org.mockito.stubbing.Stubber;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MockitoCore_ESTest extends MockitoCore_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      MockSettingsImpl mockSettingsImpl0 = new MockSettingsImpl();
      // Undeclared exception!
      try { 
        mockitoCore0.stubVoid(mockSettingsImpl0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.stub("'");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Class<Invocation> class0 = Invocation.class;
      // Undeclared exception!
      try { 
        mockitoCore0.mock(class0, (MockSettings) null, false);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.util.MockUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Stubber stubber0 = mockitoCore0.doAnswer((Answer) null);
      assertNotNull(stubber0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.when((Object) mockitoCore0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.getLastInvocation();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.MockitoCore", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      mockitoCore0.validateMockitoUsage();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.stub();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Only only0 = new Only();
      // Undeclared exception!
      try { 
        mockitoCore0.verify((Object) mockitoCore0, (VerificationMode) only0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.verify((Object) null, (VerificationMode) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      MockSettingsImpl[] mockSettingsImplArray0 = new MockSettingsImpl[0];
      mockitoCore0.reset(mockSettingsImplArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Object[] objectArray0 = new Object[1];
      // Undeclared exception!
      try { 
        mockitoCore0.reset(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Object[] objectArray0 = new Object[1];
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.verifyNoMoreInteractions(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Object[] objectArray0 = new Object[1];
      objectArray0[0] = (Object) mockitoCore0;
      // Undeclared exception!
      try { 
        mockitoCore0.verifyNoMoreInteractions(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.verifyNoMoreInteractions((Object[]) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Object[] objectArray0 = new Object[0];
      // Undeclared exception!
      try { 
        mockitoCore0.verifyNoMoreInteractions(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.inOrder((Object[]) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Object[] objectArray0 = new Object[8];
      // Undeclared exception!
      try { 
        mockitoCore0.inOrder(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Object[] objectArray0 = new Object[0];
      // Undeclared exception!
      try { 
        mockitoCore0.inOrder(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Object[] objectArray0 = new Object[8];
      objectArray0[0] = (Object) mockitoCore0;
      // Undeclared exception!
      try { 
        mockitoCore0.inOrder(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}
