/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:48:13 GMT 2023
 */

package org.apache.commons.lang3.text.translate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.CharBuffer;
import org.apache.commons.lang3.text.translate.NumericEntityUnescaper;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class NumericEntityUnescaper_ESTest extends NumericEntityUnescaper_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      char[] charArray0 = new char[8];
      charArray0[2] = '&';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      String string0 = numericEntityUnescaper0.translate((CharSequence) charBuffer0);
      assertEquals("\u0000\u0000&\u0000\u0000\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      char[] charArray0 = new char[1];
      charArray0[0] = '&';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      String string0 = numericEntityUnescaper0.translate((CharSequence) charBuffer0);
      assertEquals("&", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      char[] charArray0 = new char[9];
      charArray0[1] = '#';
      charArray0[2] = 'X';
      charArray0[0] = '&';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      // Undeclared exception!
      try { 
        numericEntityUnescaper0.translate((CharSequence) charBuffer0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.nio.Buffer", e);
      }
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      char[] charArray0 = new char[9];
      charArray0[0] = '&';
      charArray0[1] = '#';
      charArray0[2] = 'x';
      charArray0[6] = ';';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      String string0 = numericEntityUnescaper0.translate((CharSequence) charBuffer0);
      assertEquals("&#x\u0000\u0000\u0000;\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      NumericEntityUnescaper numericEntityUnescaper0 = new NumericEntityUnescaper();
      char[] charArray0 = new char[8];
      charArray0[2] = '&';
      charArray0[3] = '#';
      charArray0[4] = '1';
      charArray0[5] = ';';
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      String string0 = numericEntityUnescaper0.translate((CharSequence) charBuffer0);
      assertEquals("\u0000\u0000\u0001\u0000\u0000", string0);
  }
}
