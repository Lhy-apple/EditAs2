/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 20:07:34 GMT 2023
 */

package org.mockito.internal;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.lang.annotation.Annotation;
import java.util.LinkedList;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;
import org.mockito.MockSettings;
import org.mockito.internal.MockitoCore;
import org.mockito.internal.creation.MockSettingsImpl;
import org.mockito.internal.verification.InOrderContextImpl;
import org.mockito.internal.verification.NoMoreInteractions;
import org.mockito.stubbing.Answer;
import org.mockito.verification.VerificationMode;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class MockitoCore_ESTest extends MockitoCore_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.stubVoid("Z;kB-0BOv17\"_");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.stub("Rg*2LmjdB!Yz\"%N");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      mockitoCore0.doAnswer((Answer) null);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      LinkedList<Object> linkedList0 = new LinkedList<Object>();
      InOrderContextImpl inOrderContextImpl0 = new InOrderContextImpl();
      // Undeclared exception!
      try { 
        mockitoCore0.verifyNoMoreInteractionsInOrder(linkedList0, inOrderContextImpl0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Class<Object> class0 = Object.class;
      MockSettingsImpl mockSettingsImpl0 = new MockSettingsImpl();
      // Undeclared exception!
      try { 
        mockitoCore0.mock(class0, (MockSettings) mockSettingsImpl0);
        fail("Expecting exception: IncompatibleClassChangeError");
      
      } catch(IncompatibleClassChangeError e) {
         //
         // Expected non-static field org.mockito.cglib.proxy.Enhancer.serialVersionUID
         //
         verifyException("org.mockito.cglib.proxy.Enhancer", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      // Undeclared exception!
      try { 
        mockitoCore0.when("Multiple callback type possible but no filter specified");
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
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
  public void test07()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      mockitoCore0.validateMockitoUsage();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      NoMoreInteractions noMoreInteractions0 = new NoMoreInteractions();
      // Undeclared exception!
      try { 
        mockitoCore0.verify((Object) mockitoCore0, (VerificationMode) noMoreInteractions0);
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
      Annotation[] annotationArray0 = new Annotation[0];
      mockitoCore0.reset(annotationArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Annotation[] annotationArray0 = new Annotation[1];
      // Undeclared exception!
      try { 
        mockitoCore0.reset(annotationArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      MockitoCore mockitoCore0 = new MockitoCore();
      Object[] objectArray0 = new Object[9];
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
      Object[] objectArray0 = new Object[9];
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
      Object[] objectArray0 = new Object[1];
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
      Object[] objectArray0 = new Object[1];
      objectArray0[0] = (Object) mockitoCore0;
      // Undeclared exception!
      try { 
        mockitoCore0.inOrder(objectArray0);
        fail("Expecting exception: RuntimeException");
      
      } catch(RuntimeException e) {
      }
  }
}