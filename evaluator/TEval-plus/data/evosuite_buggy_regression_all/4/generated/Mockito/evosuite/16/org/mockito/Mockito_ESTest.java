/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:17:20 GMT 2023
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
import org.mockito.internal.verification.AtMost;
import org.mockito.internal.verification.api.VerificationMode;
import org.mockito.stubbing.Answer;
import org.mockito.stubbing.ClonesArguments;
import org.mockito.stubbing.Stubber;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Mockito_ESTest extends Mockito_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Class<String> class0 = String.class;
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
      // Undeclared exception!
      try { 
        Mockito.when((Object) "");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Object object0 = new Object();
      AtMost atMost0 = new AtMost(0);
      // Undeclared exception!
      try { 
        Mockito.verify(object0, (VerificationMode) atMost0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.verifyZeroInteractions((Object[]) null);
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
        Mockito.mock(class0, ",V9D8Qz!>fH]P");
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
      assertNotNull(verificationMode0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Object[] objectArray0 = new Object[1];
      // Undeclared exception!
      try { 
        Mockito.inOrder(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.spy("org.mockito.Mockito");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
  public void test14()  throws Throwable  {
      Mockito mockito0 = new Mockito();
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Stubber stubber0 = Mockito.doAnswer((Answer) null);
      assertNotNull(stubber0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.atMost((-1809));
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Integer integer0 = new Integer((-204));
      // Undeclared exception!
      try { 
        Mockito.stubVoid((Object) integer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Integer integer0 = new Integer(0);
      // Undeclared exception!
      try { 
        Mockito.stub((Object) integer0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.verifyNoMoreInteractions((Object[]) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      VerificationMode verificationMode0 = Mockito.atLeastOnce();
      assertNotNull(verificationMode0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.atLeast((-590));
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Object object0 = new Object();
      Stubber stubber0 = Mockito.doReturn(object0);
      assertNotNull(stubber0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.verify((Object) null);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      // Undeclared exception!
      try { 
        Mockito.validateMockitoUsage();
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Stubber stubber0 = Mockito.doThrow((Throwable) null);
      assertNotNull(stubber0);
  }
}
