/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 18:30:18 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.apache.commons.codec.binary.Base32;
import org.apache.commons.codec.binary.BaseNCodec;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base32_ESTest extends Base32_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base32 base32_0 = new Base32(4, byteArray0, false);
      assertEquals(76, BaseNCodec.MIME_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Base32 base32_0 = new Base32((byte) (-10));
      assertEquals(64, BaseNCodec.PEM_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Base32 base32_0 = new Base32((-876), (byte[]) null);
      Object object0 = base32_0.decode((Object) "n,:#7=");
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Base32 base32_0 = new Base32((-2375));
      byte[] byteArray0 = base32_0.decode("rWW7`x:RK6#3@p");
      assertEquals(4, byteArray0.length);
      assertArrayEquals(new byte[] {(byte) (-75), (byte) (-65), (byte)21, (byte)123}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Base32 base32_0 = new Base32();
      Object object0 = base32_0.decode((Object) "\u0002HPFARh5jk!|u~NCQ");
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(39, (byte[]) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineLength 39 > 0, but lineSeparator is null
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)13;
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(67, byteArray0, true, (byte)13);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeparator must not contain Base32 characters: [\r]
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(true, (byte)52);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // pad must not be in alphabet or whitespace
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(true, (byte)32);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // pad must not be in alphabet or whitespace
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Base32 base32_0 = new Base32(8192);
      Object object0 = base32_0.decode((Object) "$(Ag(Ht(!~V9*#");
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Base32 base32_0 = new Base32(8192);
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      byte[] byteArray0 = new byte[2];
      base32_0.encode(byteArray0, (-31), (-241), baseNCodec_Context0);
      base32_0.decode(byteArray0, 25, (-1142), baseNCodec_Context0);
      assertEquals(76, BaseNCodec.MIME_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Base32 base32_0 = new Base32((-876), (byte[]) null);
      Object object0 = base32_0.decode((Object) "lineSeparator must not contain Base32 characters: [");
      Object object1 = base32_0.decode(object0);
      assertNotSame(object1, object0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Base32 base32_0 = new Base32((-876), (byte[]) null);
      byte[] byteArray0 = base32_0.decode("!27@HAfdyD0%");
      assertArrayEquals(new byte[] {(byte) (-41), (byte) (-50), (byte)1}, byteArray0);
      assertEquals(3, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = base32_0.decode("GcU)1XmZf3Z");
      String string0 = base32_0.encodeToString(byteArray0);
      assertEquals("GUXZ2===", string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Base32 base32_0 = new Base32((-876), (byte[]) null);
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      byte[] byteArray0 = base32_0.decode("lineSeparator must not contain Base32 characters: [");
      baseNCodec_Context0.modulus = 1788;
      // Undeclared exception!
      try { 
        base32_0.decode(byteArray0, (-1900), (-125), baseNCodec_Context0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Impossible modulus 1788
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Base32 base32_0 = new Base32((-2375));
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      base32_0.encode((byte[]) null, 78, (-32), baseNCodec_Context0);
      base32_0.encode((byte[]) null, (-1), 0, baseNCodec_Context0);
      assertEquals(76, BaseNCodec.MIME_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Base32 base32_0 = new Base32((-876), (byte[]) null);
      byte[] byteArray0 = base32_0.decode("lineSeparator must not contain Base32 characters: [");
      String string0 = base32_0.encodeToString(byteArray0);
      assertEquals("SB3Q====", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Base32 base32_0 = new Base32(true, (byte) (-122));
      byte[] byteArray0 = new byte[4];
      String string0 = base32_0.encodeToString(byteArray0);
      assertEquals("0000000\uFFFD", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = new byte[3];
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      baseNCodec_Context0.modulus = (-1228);
      // Undeclared exception!
      try { 
        base32_0.encode(byteArray0, 0, (int) (byte) (-91), baseNCodec_Context0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Impossible modulus -1228
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Base32 base32_0 = new Base32(59);
      byte[] byteArray0 = new byte[16];
      String string0 = base32_0.encodeAsString(byteArray0);
      assertEquals("AAAAAAAAAAAAAAAAAAAAAAAAAA======\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Base32 base32_0 = new Base32(true, (byte) (-122));
      byte[] byteArray0 = new byte[6];
      String string0 = base32_0.encodeAsString(byteArray0);
      assertEquals("0000000000\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base32 base32_0 = new Base32(12, byteArray0);
      byte[] byteArray1 = new byte[21];
      String string0 = base32_0.encodeAsString(byteArray1);
      assertEquals("AAAAAAAA\u0000\u0000\u0000AAAAAAAA\u0000\u0000\u0000AAAAAAAA\u0000\u0000\u0000AAAAAAAA\u0000\u0000\u0000AA======\u0000\u0000\u0000", string0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Base32 base32_0 = new Base32((-876));
      boolean boolean0 = base32_0.isInAlphabet("KKwWt_`");
      assertFalse(boolean0);
  }
}