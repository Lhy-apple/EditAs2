/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:57:45 GMT 2023
 */

package org.apache.commons.lang3.text.translate;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.CharBuffer;
import org.apache.commons.lang3.text.translate.CharSequenceTranslator;
import org.apache.commons.lang3.text.translate.NumericEntityEscaper;
import org.apache.commons.lang3.text.translate.OctalUnescaper;
import org.apache.commons.lang3.text.translate.UnicodeEscaper;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class CharSequenceTranslator_ESTest extends CharSequenceTranslator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      OctalUnescaper octalUnescaper0 = new OctalUnescaper();
      CharSequenceTranslator[] charSequenceTranslatorArray0 = new CharSequenceTranslator[8];
      CharSequenceTranslator charSequenceTranslator0 = octalUnescaper0.with(charSequenceTranslatorArray0);
      assertNotNull(charSequenceTranslator0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      UnicodeEscaper unicodeEscaper0 = UnicodeEscaper.outsideOf((-3138), (-3138));
      char[] charArray0 = new char[8];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      String string0 = unicodeEscaper0.translate((CharSequence) charBuffer0);
      assertEquals("\\u0000\\u0000\\u0000\\u0000\\u0000\\u0000\\u0000\\u0000", string0);
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      NumericEntityEscaper numericEntityEscaper0 = NumericEntityEscaper.below(0);
      String string0 = numericEntityEscaper0.translate((CharSequence) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      UnicodeEscaper unicodeEscaper0 = UnicodeEscaper.above(16);
      CharBuffer charBuffer0 = CharBuffer.allocate(16);
      // Undeclared exception!
      try { 
        unicodeEscaper0.translate((CharSequence) charBuffer0, (Writer) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The Writer must not be null
         //
         verifyException("org.apache.commons.lang3.text.translate.CharSequenceTranslator", e);
      }
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      UnicodeEscaper unicodeEscaper0 = UnicodeEscaper.above(2418);
      StringWriter stringWriter0 = new StringWriter();
      unicodeEscaper0.translate((CharSequence) null, (Writer) stringWriter0);
      assertEquals("", stringWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      UnicodeEscaper unicodeEscaper0 = UnicodeEscaper.above(65);
      CharBuffer charBuffer0 = CharBuffer.allocate(65);
      StringWriter stringWriter0 = new StringWriter(65);
      unicodeEscaper0.translate((CharSequence) charBuffer0, (Writer) stringWriter0);
      assertEquals("\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", stringWriter0.toString());
  }
}
