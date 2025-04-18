/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:09:37 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigInteger;
import org.apache.commons.codec.binary.Base64;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base64_ESTest extends Base64_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      String string0 = Base64.encodeBase64String(byteArray0);
      assertEquals("DQo=\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Base64 base64_0 = new Base64();
      boolean boolean0 = base64_0.hasData();
      assertFalse(boolean0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("7rLqyg");
      assertEquals(4, byteArray0.length);
      assertArrayEquals(new byte[] {(byte) (-18), (byte) (-78), (byte) (-22), (byte) (-54)}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("DQo", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)65}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = Base64.encodeBase64((byte[]) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = Base64.encodeBase64Chunked((byte[]) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64.decodeBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      Base64 base64_0 = new Base64(1858, byteArray0, false);
      Object object0 = base64_0.decode((Object) "or.apachecommons.codcEncoderException");
      base64_0.encode(object0);
      boolean boolean0 = base64_0.hasData();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.avail();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64();
      base64_0.readResults(byteArray0, 97, (-296));
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64(2169);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      base64_0.readResults(byteArray1, 16, 16);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64(5, byteArray0);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      base64_0.encode(byteArray1);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64();
      base64_0.decode(byteArray0);
      base64_0.readResults(byteArray0, 97, (-296));
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Base64 base64_0 = new Base64(92, (byte[]) null, true);
      base64_0.setInitialBuffer((byte[]) null, (byte)65, 92);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      Base64 base64_0 = new Base64(1131, byteArray0);
      base64_0.setInitialBuffer(byteArray0, 1345, (-3729));
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64(10, byteArray0);
      base64_0.decode(byteArray0);
      base64_0.encode(byteArray0, 10, 10);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64(false);
      base64_0.encode(byteArray0, 1453, (-1460));
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      Base64 base64_0 = new Base64(2238, byteArray0);
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray0, 51, 2238);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 51
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Base64 base64_0 = new Base64(1);
      byte[] byteArray0 = new byte[1];
      byte[] byteArray1 = base64_0.encode(byteArray0);
      base64_0.encode(byteArray1);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Base64 base64_0 = new Base64();
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      // Undeclared exception!
      try { 
        base64_0.decode(byteArray0, 64, 64);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 64
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[0] = (byte) (-1);
      Base64 base64_0 = new Base64(1131, byteArray0);
      byte[] byteArray1 = base64_0.decode(byteArray0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      Base64 base64_0 = new Base64(1858, byteArray0, false);
      Object object0 = base64_0.decode((Object) "org.apache.commons.codec.EncoderException");
      assertNotSame(byteArray0, object0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64.isArrayByteBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte)123;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte)32;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Base64 base64_0 = new Base64();
      Object object0 = base64_0.decode((Object) "UTF-16BE");
      base64_0.decode(object0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Base64 base64_0 = new Base64();
      Object object0 = new Object();
      try { 
        base64_0.decode(object0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 decode is not a byte[] or a String
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Base64 base64_0 = new Base64();
      base64_0.decode((byte[]) null);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      Base64 base64_0 = new Base64(1131, byteArray0);
      byte[] byteArray1 = base64_0.decode(byteArray0);
      Base64.decodeInteger(byteArray1);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertNotSame(byteArray1, byteArray0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, true, true, (int) (byte) (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (10) than the specified maxium size of -1
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte)9;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)9, (byte)0}, byteArray0);
      assertEquals(1, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[2] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0}, byteArray1);
      assertEquals(3, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)32, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[0] = (byte)75;
      byteArray0[1] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      Base64 base64_0 = new Base64(1858, byteArray0, false);
      try { 
        base64_0.encode((Object) "orapachecommonscodcEncoderExceptiok=");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Base64 base64_0 = new Base64();
      String string0 = base64_0.encodeToString((byte[]) null);
      assertNull(string0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Base64 base64_0 = new Base64((-1));
      byte[] byteArray0 = new byte[0];
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertSame(byteArray1, byteArray0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0, true, true);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65, (byte)0, (byte)0}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      // Undeclared exception!
      try { 
        Base64.encodeInteger((BigInteger) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // encodeInteger called with null parameter
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }
}
