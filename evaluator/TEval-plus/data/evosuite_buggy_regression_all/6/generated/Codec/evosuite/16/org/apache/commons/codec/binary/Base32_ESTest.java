/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:09:08 GMT 2023
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
      byte[] byteArray0 = new byte[0];
      Base32 base32_0 = new Base32(0, byteArray0, false);
      byte[] byteArray1 = base32_0.decode("LP/|zVMcSP!3`6t");
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      baseNCodec_Context0.modulus = (int) (byte)5;
      // Undeclared exception!
      try { 
        base32_0.encode(byteArray1, (int) (byte)5, (int) (byte) (-77), baseNCodec_Context0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Impossible modulus 5
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Base32 base32_0 = new Base32(0);
      Object object0 = base32_0.decode((Object) "US-ASCII");
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = base32_0.decode("1;!E(o31Jd=E,");
      assertEquals(1, byteArray0.length);
      assertArrayEquals(new byte[] {(byte)38}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Base32 base32_0 = new Base32(true, (byte)127);
      byte[] byteArray0 = new byte[1];
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      baseNCodec_Context0.modulus = 807;
      // Undeclared exception!
      try { 
        base32_0.decode(byteArray0, (int) (byte)127, (-3931), baseNCodec_Context0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Impossible modulus 807
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base32 base32_0 = new Base32(123, byteArray0, true);
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      base32_0.encode((byte[]) null, 102, (int) (byte) (-77), baseNCodec_Context0);
      base32_0.decode((byte[]) null, (int) (byte) (-23), (-820), baseNCodec_Context0);
      assertEquals(76, BaseNCodec.MIME_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(64, (byte[]) null, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineLength 64 > 0, but lineSeparator is null
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[1] = (byte)53;
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32((byte)52, byteArray0, false);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeparator must not contain Base32 characters: [\u00005\u0000\u0000\u0000]
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32((byte)54);
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
      byte[] byteArray0 = new byte[0];
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32((-1572), byteArray0, false, (byte)13);
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
      Base32 base32_0 = new Base32((-3931));
      byte[] byteArray0 = base32_0.decode("3_&:G2m/JTi");
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      base32_0.decode(byteArray0, 1, 1, baseNCodec_Context0);
      assertArrayEquals(new byte[] {(byte) (-39), (byte) (-76), (byte) (-103)}, byteArray0);
      assertEquals(3, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Base32 base32_0 = new Base32();
      Object object0 = base32_0.decode((Object) "9zoMXG`~ZjIBK3");
      assertNotNull(object0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = base32_0.decode("t_(|0Y>/k:2");
      assertArrayEquals(new byte[] {(byte) (-58)}, byteArray0);
      assertEquals(1, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = base32_0.decode("36Ef`n8{}peVq");
      assertEquals(2, byteArray0.length);
      assertArrayEquals(new byte[] {(byte) (-33), (byte) (-119)}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = base32_0.decode("Lnun)T}4A*JD:aa");
      assertArrayEquals(new byte[] {(byte)92, (byte) (-8), (byte)4}, byteArray0);
      assertEquals(3, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Base32 base32_0 = new Base32((-1970));
      byte[] byteArray0 = new byte[5];
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      base32_0.decode(byteArray0, (-1970), (-1970), baseNCodec_Context0);
      base32_0.encode(byteArray0, (-3302), 1745, baseNCodec_Context0);
      assertEquals(64, BaseNCodec.PEM_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Base32 base32_0 = new Base32((-1970));
      byte[] byteArray0 = new byte[5];
      byte[] byteArray1 = base32_0.encode(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = new byte[17];
      String string0 = base32_0.encodeToString(byteArray0);
      assertEquals("AAAAAAAAAAAAAAAAAAAAAAAAAAAA====", string0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base32 base32_0 = new Base32((-3931));
      byte[] byteArray1 = base32_0.encode(byteArray0);
      String string0 = base32_0.encodeToString(byteArray1);
      assertEquals("IFAT2PJ5HU6T2===", string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Base32 base32_0 = new Base32(0);
      byte[] byteArray0 = new byte[4];
      String string0 = base32_0.encodeToString(byteArray0);
      assertEquals("AAAAAAA=", string0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte) (-110);
      Base32 base32_0 = new Base32(13, byteArray0, false, (byte) (-39));
      byte[] byteArray1 = base32_0.encode(byteArray0);
      assertEquals(13, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base32 base32_0 = new Base32(101);
      byte[] byteArray1 = base32_0.encode(byteArray0);
      String string0 = base32_0.encodeToString(byteArray1);
      assertEquals("IFAT2PJ5HU6T2DIK\r\n", string0);
  }
}