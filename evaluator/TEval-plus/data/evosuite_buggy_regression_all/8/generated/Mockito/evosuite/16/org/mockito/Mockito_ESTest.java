/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 04:17:58 GMT 2023
 */

package org.mockito;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.mockito.Mockito;
import org.mockito.MockitoDebugger;
import org.mockito.internal.verification.api.VerificationMode;
import org.mockito.stubbing.Answer;
import org.mockito.stubbing.ClonesArguments;
import org.mockito.stubbing.Stubber;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Mockito_ESTest extends Mockito_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        Mockito.mock(class0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Stubber stubber0 = Mockito.doNothing();
      assertNotNull(stubber0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      VerificationMode verificationMode0 = Mockito.only();
      assertNotNull(verificationMode0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Integer integer0 = new Integer(1);
      // Undeclared exception!
      try { 
        Mockito.when(integer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      VerificationMode verificationMode0 = Mockito.atLeastOnce();
      // Undeclared exception!
      try { 
        Mockito.verify(" but was ", verificationMode0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      String[] stringArray0 = new String[8];
      // Undeclared exception!
      try { 
        Mockito.verifyZeroInteractions(stringArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.reset((String[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.mockito.internal.MockitoCore", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Stubber stubber0 = Mockito.doCallRealMethod();
      assertNotNull(stubber0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Class<Object> class0 = Object.class;
      // Undeclared exception!
      try { 
        Mockito.mock(class0, "org.mockito.Mockito");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      MockitoDebugger mockitoDebugger0 = Mockito.debug();
      assertNotNull(mockitoDebugger0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      VerificationMode verificationMode0 = Mockito.never();
      // Undeclared exception!
      try { 
        Mockito.spy((Object) verificationMode0);
        fail("Expecting exception: IncompatibleClassChangeError");
      
      } catch(IncompatibleClassChangeError e) {
         //
         // Expected non-static field org.mockito.cglib.proxy.Enhancer.serialVersionUID
         //
         verifyException("org.mockito.cglib.proxy.Enhancer", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Object[] objectArray0 = new Object[3];
      // Undeclared exception!
      try { 
        Mockito.inOrder(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Class<Object> class0 = Object.class;
      ClonesArguments clonesArguments0 = new ClonesArguments();
      // Undeclared exception!
      try { 
        Mockito.mock(class0, (Answer) clonesArguments0);
        fail("Expecting exception: IncompatibleClassChangeError");
      
      } catch(IncompatibleClassChangeError e) {
         //
         // Expected non-static field org.mockito.cglib.proxy.Enhancer.serialVersionUID
         //
         verifyException("org.mockito.cglib.proxy.Enhancer", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Mockito mockito0 = new Mockito();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Stubber stubber0 = Mockito.doAnswer((Answer) null);
      assertNotNull(stubber0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      VerificationMode verificationMode0 = Mockito.atMost(1953);
      assertNotNull(verificationMode0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Integer integer0 = new Integer(86);
      // Undeclared exception!
      try { 
        Mockito.stubVoid((Object) integer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Integer integer0 = new Integer(26);
      // Undeclared exception!
      try { 
        Mockito.stub(integer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Object[] objectArray0 = new Object[3];
      // Undeclared exception!
      try { 
        Mockito.verifyNoMoreInteractions(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.atLeast((-1));
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Stubber stubber0 = Mockito.doReturn((Object) null);
      assertNotNull(stubber0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.verify((Object) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.validateMockitoUsage();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Stubber stubber0 = Mockito.doThrow((Throwable) null);
      assertNotNull(stubber0);
  }
}