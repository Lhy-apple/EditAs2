/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:58:39 GMT 2023
 */

package org.mockito.internal.matchers;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.hamcrest.StringDescription;
import org.junit.runner.RunWith;
import org.mockito.internal.matchers.Same;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Same_ESTest extends Same_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Same same0 = new Same((Object) null);
      StringDescription stringDescription0 = new StringDescription();
      // Undeclared exception!
      try { 
        same0.describeTo(stringDescription0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.evosuite.runtime.System", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Same same0 = new Same("O?dss/kyB#");
      boolean boolean0 = same0.matches("O?dss/kyB#");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      Same same0 = new Same("O?dss/kyB#");
      String string0 = same0.toString();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Character character0 = new Character('\"');
      Same same0 = new Same(character0);
      String string0 = same0.toString();
      assertNotNull(string0);
  }
}
