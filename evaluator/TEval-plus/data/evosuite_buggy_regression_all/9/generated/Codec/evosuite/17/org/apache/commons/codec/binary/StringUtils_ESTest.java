/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:17:23 GMT 2023
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
      byte[] byteArray0 = StringUtils.getBytesUtf16Be("");
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      // Undeclared exception!
      try { 
        StringUtils.getBytesUnchecked("", "");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // : java.io.UnsupportedEncodingException: 
         //
         verifyException("org.apache.commons.codec.binary.StringUtils", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf16("P_zu ]eA:wWn");
      String string0 = StringUtils.newStringUsAscii(byteArray0);
      assertEquals("\uFFFD\uFFFD\u0000P\u0000_\u0000z\u0000u\u0000 \u0000]\u0000e\u0000A\u0000:\u0000w\u0000W\u0000n", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      String string0 = StringUtils.newStringIso8859_1(byteArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUsAscii("\u4400\u2B00\u7B00\u2E00\u5D00\u7900\u5100");
      assertEquals(7, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      String string0 = StringUtils.newStringUtf16Le(byteArray0);
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf16Le("D+{.]yQ");
      String string0 = StringUtils.newStringUtf16Be(byteArray0);
      assertEquals("\u4400\u2B00\u7B00\u2E00\u5D00\u7900\u5100", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      String string0 = StringUtils.newStringUtf16((byte[]) null);
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      StringUtils stringUtils0 = new StringUtils();
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesIso8859_1("jT+X_r;mD<Hp+et#$");
      assertEquals(17, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf8((String) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      ByteBuffer byteBuffer0 = StringUtils.getByteBufferUtf8("jT+X_r;mD<Hp+et#$");
      assertEquals(0, byteBuffer0.position());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf16Le("D+{.]yQ");
      String string0 = StringUtils.newStringUtf8(byteArray0);
      assertEquals("D\u0000+\u0000{\u0000.\u0000]\u0000y\u0000Q\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) null, (CharSequence) "jT+X_r;mD<Hp+et#$");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) "\u0100\u0001\u0001", (CharSequence) "\u0100\u0001\u0001");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) "", (CharSequence) "D\u0000+\u0000{\u0000.\u0000]\u0000y\u0000Q\u0000");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      boolean boolean0 = StringUtils.equals((CharSequence) "UTF-16BE", (CharSequence) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      char[] charArray0 = new char[3];
      CharBuffer charBuffer0 = CharBuffer.wrap(charArray0, 1, 0);
      // Undeclared exception!
      try { 
        StringUtils.equals((CharSequence) charBuffer0, (CharSequence) "\uFFFD\uFFFD\u0000o\u0000J\u0000Y\u0000P\u0000\u0000*\u0000r\u0000Y\u0000.\u0000b\u0000_\u0000J\u0000K\u0000%");
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.nio.Buffer", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      CharBuffer charBuffer0 = CharBuffer.wrap((CharSequence) "||dZ|o^~G+l%# Cp,=", 1, 1);
      boolean boolean0 = StringUtils.equals((CharSequence) "", (CharSequence) charBuffer0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      ByteBuffer byteBuffer0 = StringUtils.getByteBufferUtf8((String) null);
      assertNull(byteBuffer0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUnchecked((String) null, (String) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = StringUtils.getBytesUtf16("qX0*t*K*G,?=");
      String string0 = StringUtils.newString(byteArray0, "US-ASCII");
      assertEquals("\uFFFD\uFFFD\u0000q\u0000X\u00000\u0000*\u0000t\u0000*\u0000K\u0000*\u0000G\u0000,\u0000?\u0000=", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      String string0 = StringUtils.newString((byte[]) null, "M");
      assertNull(string0);
  }
}