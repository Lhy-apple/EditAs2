/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 02:37:47 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import org.apache.commons.codec.binary.StringUtils;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class StringUtils_ESTest extends StringUtils_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf16Be("NsH(,");
      assertArrayEquals(new byte[] {(byte)0, (byte)78, (byte)0, (byte)115, (byte)0, (byte)72, (byte)0, (byte)40, (byte)0, (byte)44}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringUtils.getBytesUnchecked(": ", "3NnmjeA");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // 3NnmjeA: java.io.UnsupportedEncodingException: 3NnmjeA
         //
         verifyException("org.apache.commons.codec.binary.StringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf8(": ");
      String string0 = StringUtils.newStringUsAscii(byteArray0);
      assertEquals(": ", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      String string0 = StringUtils.newStringIso8859_1(byteArray0);
      assertEquals("\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUsAscii(": ");
      String string0 = StringUtils.newStringUtf16Le(byteArray0);
      assertEquals("\u203A", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      String string0 = StringUtils.newStringUtf16Be(byteArray0);
      assertEquals("\u0000\u0000\u0000\uFFFD", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf16Le("PAsCu}{}%%WZ!|");
      assertEquals(28, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      StringUtils stringUtils0 = new StringUtils();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      ByteBuffer byteBuffer0 = StringUtils.getByteBufferUtf8("fJ0iU>[");
      assertTrue(byteBuffer0.hasRemaining());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf16(": ");
      assertEquals(6, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf8(": ");
      String string0 = StringUtils.newStringUtf8(byteArray0);
      assertEquals(": ", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) "XRk]!%3D3CQkU;", (CharSequence) "");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) "", (CharSequence) "");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) null, (CharSequence) "UTF-8");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) "UTF-8", (CharSequence) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      CharBuffer charBuffer0 = CharBuffer.wrap((CharSequence) "\u7900)\uFF20\uFFFD");
      boolean boolean0 = StringUtils.equals((CharSequence) charBuffer0, (CharSequence) "\u7900)\uFF20\uFFFD");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      char[] charArray0 = new char[4];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0, 0, 0);
      // Undeclared exception!
      try { 
        StringUtils.equals((CharSequence) "A48S seXcUTr6(9&v", (CharSequence) charBuffer0);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.nio.Buffer", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesIso8859_1((String) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      ByteBuffer byteBuffer0 = StringUtils.getByteBufferUtf8((String) null);
      assertNull(byteBuffer0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUnchecked((String) null, "A48S seXcUTr6(9&v");
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      String string0 = StringUtils.newStringUtf16((byte[]) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      // Undeclared exception!
      try { 
        StringUtils.newString(byteArray0, (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = StringUtils.newString((byte[]) null, "UTF-8");
      assertNull(string0);
  }
}