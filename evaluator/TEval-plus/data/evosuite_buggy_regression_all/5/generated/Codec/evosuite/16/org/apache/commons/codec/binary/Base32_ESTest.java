/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:47:43 GMT 2023
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
      Base32 base32_0 = new Base32((byte)0);
      assertEquals(64, BaseNCodec.PEM_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base32 base32_0 = new Base32(8192, byteArray0);
      byte[] byteArray1 = base32_0.decode("7.Wu-4\"xMs.29K}n9^");
      assertEquals(3, byteArray1.length);
      assertArrayEquals(new byte[] {(byte) (-3), (byte) (-72), (byte) (-51)}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = new byte[4];
      byteArray0[0] = (byte) (-112);
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      // Undeclared exception!
      try { 
        base32_0.decode(byteArray0, (int) (byte)0, 1048, baseNCodec_Context0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 4
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(41, (byte[]) null, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineLength 41 > 0, but lineSeparator is null
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[1] = (byte)22;
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(64, byteArray0, false, (byte)22);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeparator must not contain Base32 characters: [\u0000\u0016\u0000]
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(true, (byte)76);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // pad must not be in alphabet or whitespace
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Base32 base32_0 = null;
      try {
        base32_0 = new Base32(true, (byte)10);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // pad must not be in alphabet or whitespace
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base32 base32_0 = new Base32(true, (byte)0);
      byte[] byteArray0 = new byte[3];
      byte[] byteArray1 = base32_0.decode(byteArray0);
      assertNotSame(byteArray0, byteArray1);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = base32_0.decode("s3:6NA\u0007h/s&KxGaLq");
      assertEquals(4, byteArray0.length);
      assertArrayEquals(new byte[] {(byte) (-33), (byte) (-102), (byte)5, (byte)25}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      Base32 base32_0 = new Base32((byte) (-16), byteArray0);
      byte[] byteArray1 = base32_0.decode("N6");
      assertEquals(1, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)111}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = base32_0.decode("4Qj@qH+");
      assertArrayEquals(new byte[] {(byte) (-28)}, byteArray0);
      assertEquals(1, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base32 base32_0 = new Base32(8192, byteArray0);
      byte[] byteArray1 = base32_0.decode("p{UI~ AOd");
      assertArrayEquals(new byte[] {(byte) (-94), (byte)0}, byteArray1);
      assertEquals(2, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base32 base32_0 = new Base32(574, byteArray0);
      byte[] byteArray1 = base32_0.decode("%6[jDZ!i60)(!x>rP");
      assertArrayEquals(new byte[] {(byte) (-16), (byte) (-13), (byte) (-25)}, byteArray1);
      assertEquals(3, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = new byte[2];
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      baseNCodec_Context0.modulus = 76;
      // Undeclared exception!
      try { 
        base32_0.decode(byteArray0, 32, (-2085), baseNCodec_Context0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Impossible modulus 76
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base32 base32_0 = new Base32(8192, byteArray0);
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      base32_0.encode(byteArray0, 80, (-6634), baseNCodec_Context0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = new byte[4];
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      base32_0.decode(byteArray0, 89, (-30), baseNCodec_Context0);
      base32_0.encode(byteArray0, (int) (byte) (-112), (-30), baseNCodec_Context0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base32 base32_0 = new Base32(8192, byteArray0);
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      // Undeclared exception!
      try { 
        base32_0.encode(byteArray0, 0, 1918, baseNCodec_Context0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 7
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base32 base32_0 = new Base32(8192, byteArray0);
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      baseNCodec_Context0.modulus = 8;
      // Undeclared exception!
      try { 
        base32_0.encode(byteArray0, 80, (-6634), baseNCodec_Context0);
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Impossible modulus 8
         //
         verifyException("org.apache.commons.codec.binary.Base32", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = new byte[4];
      BaseNCodec.Context baseNCodec_Context0 = new BaseNCodec.Context();
      base32_0.encode(byteArray0, (-1), (-1), baseNCodec_Context0);
      assertEquals(64, BaseNCodec.PEM_CHUNK_SIZE);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Base32 base32_0 = new Base32(true, (byte) (-103));
      byte[] byteArray0 = new byte[1];
      byte[] byteArray1 = base32_0.encode(byteArray0);
      assertArrayEquals(new byte[] {(byte)48, (byte)48, (byte) (-103), (byte) (-103), (byte) (-103), (byte) (-103), (byte) (-103), (byte) (-103)}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Base32 base32_0 = new Base32(true, (byte)0);
      byte[] byteArray0 = new byte[3];
      byte[] byteArray1 = base32_0.encode(byteArray0);
      assertArrayEquals(new byte[] {(byte)48, (byte)48, (byte)48, (byte)48, (byte)48, (byte)0, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Base32 base32_0 = new Base32();
      byte[] byteArray0 = new byte[4];
      byte[] byteArray1 = base32_0.encode(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)65, (byte)61}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[1] = (byte) (-16);
      Base32 base32_0 = new Base32();
      byte[] byteArray1 = base32_0.encode(byteArray0);
      assertEquals(16, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Base32 base32_0 = new Base32(8);
      byte[] byteArray0 = new byte[8];
      byte[] byteArray1 = base32_0.encode(byteArray0);
      assertEquals(20, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Base32 base32_0 = new Base32(false, (byte)91);
      assertEquals(64, BaseNCodec.PEM_CHUNK_SIZE);
  }
}
