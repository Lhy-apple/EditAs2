/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 14:20:29 GMT 2023
 */

package org.apache.commons.math3.dfp;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.math3.dfp.Dfp;
import org.apache.commons.math3.dfp.DfpField;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Dfp_ESTest extends Dfp_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      Dfp dfp0 = null;
      try {
        dfp0 = new Dfp((DfpField) null, (byte)101);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.dfp.Dfp", e);
      }
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      // Undeclared exception!
      try { 
        Dfp.copysign((Dfp) null, (Dfp) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.dfp.Dfp", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      DfpField dfpField0 = new DfpField((-477));
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      Dfp dfp0 = null;
      try {
        dfp0 = new Dfp((DfpField) null, (byte) (-62), (byte) (-62));
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.dfp.Dfp", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      Dfp dfp0 = null;
      try {
        dfp0 = new Dfp((DfpField) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.apache.commons.math3.dfp.Dfp", e);
      }
  }
}
