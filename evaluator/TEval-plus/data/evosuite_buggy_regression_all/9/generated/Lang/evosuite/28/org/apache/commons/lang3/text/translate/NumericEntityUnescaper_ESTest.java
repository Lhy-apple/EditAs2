/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:26:23 GMT 2023
 */

package org.apache.commons.lang3.text.translate;

import org.junit.Test;
import static org.junit.Assert.*;
import org.apache.commons.lang3.text.translate.NumericEntityUnescaper;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumericEntityUnescaper_ESTest extends NumericEntityUnescaper_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      String string0 = numericEntityUnescaper0.translate((CharSequence) "&fBa&#x;)Z=:");
      assertEquals("&fBa&#x;)Z=:", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      String string0 = numericEntityUnescaper0.translate((CharSequence) "+fOaU&#X;\"=:");
      assertEquals("+fOaU&#X;\"=:", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      String string0 = numericEntityUnescaper0.translate((CharSequence) "&fOa&#9;)=:");
      assertEquals("&fOa\t)=:", string0);
  }
}
