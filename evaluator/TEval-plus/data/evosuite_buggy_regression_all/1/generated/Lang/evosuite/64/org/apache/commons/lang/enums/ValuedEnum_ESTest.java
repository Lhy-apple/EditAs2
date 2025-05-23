/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 14:01:04 GMT 2023
 */

package org.apache.commons.lang.enums;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.lang.enums.ValuedEnum;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class ValuedEnum_ESTest extends ValuedEnum_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Class<Integer> class0 = Integer.class;
      // Undeclared exception!
      try { 
        ValuedEnum.getEnum(class0, 5505);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Class must be a subclass of Enum
         //
         verifyException("org.apache.commons.lang.enums.Enum", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      // Undeclared exception!
      try { 
        ValuedEnum.getEnum((Class) null, 1020);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Enum Class must not be null
         //
         verifyException("org.apache.commons.lang.enums.ValuedEnum", e);
      }
  }
}
