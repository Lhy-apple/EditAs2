/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:16:56 GMT 2023
 */

package org.apache.commons.lang3.text.translate;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.CharBuffer;
import org.apache.commons.lang3.text.translate.LookupTranslator;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class LookupTranslator_ESTest extends LookupTranslator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      LookupTranslator lookupTranslator0 = new LookupTranslator((CharSequence[][]) null);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      CharSequence[][] charSequenceArray0 = new CharSequence[0][4];
      LookupTranslator lookupTranslator0 = new LookupTranslator(charSequenceArray0);
      char[] charArray0 = new char[0];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      StringWriter stringWriter0 = new StringWriter();
      int int0 = lookupTranslator0.translate((CharSequence) charBuffer0, 2368, (Writer) stringWriter0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      CharSequence[][] charSequenceArray0 = new CharSequence[3][4];
      CharSequence[] charSequenceArray1 = new CharSequence[3];
      char[] charArray0 = new char[1];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      charSequenceArray1[0] = (CharSequence) charBuffer0;
      charSequenceArray1[1] = (CharSequence) "170";
      StringWriter stringWriter0 = new StringWriter();
      charSequenceArray0[0] = charSequenceArray1;
      charSequenceArray0[1] = charSequenceArray0[0];
      charSequenceArray0[2] = charSequenceArray0[0];
      LookupTranslator lookupTranslator0 = new LookupTranslator(charSequenceArray0);
      int int0 = lookupTranslator0.translate(charSequenceArray1[1], 0, (Writer) stringWriter0);
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      CharSequence[][] charSequenceArray0 = new CharSequence[3][4];
      CharSequence[] charSequenceArray1 = new CharSequence[3];
      char[] charArray0 = new char[1];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0);
      charSequenceArray1[0] = (CharSequence) charBuffer0;
      charSequenceArray1[1] = (CharSequence) "170";
      StringWriter stringWriter0 = new StringWriter();
      charSequenceArray0[0] = charSequenceArray1;
      charSequenceArray0[1] = charSequenceArray1;
      charSequenceArray0[2] = charSequenceArray0[0];
      LookupTranslator lookupTranslator0 = new LookupTranslator(charSequenceArray0);
      int int0 = lookupTranslator0.translate(charSequenceArray1[0], 0, (Writer) stringWriter0);
      assertEquals("170", stringWriter0.toString());
      assertEquals(1, int0);
  }
}
