/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 05:17:05 GMT 2023
 */

package org.apache.commons.codec.binary;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.math.BigInteger;
import org.apache.commons.codec.binary.Base64;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.util.MockRandom;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Base64_ESTest extends Base64_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(0, byteArray0);
      String string0 = base64_0.encodeToString(byteArray0);
      assertEquals("", string0);
      assertEquals(0, byteArray0.length);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      String string0 = Base64.encodeBase64String(byteArray0);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
      assertEquals("Q2c9PQ==\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("#JFEWQ8Rwk");
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      assertEquals(8, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)36, (byte)81, (byte)22, (byte)67, (byte) (-60), (byte)112}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      BigInteger bigInteger0 = Base64.decodeInteger(byteArray0);
      assertEquals((short)0, bigInteger0.shortValue());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertArrayEquals(new byte[] {(byte)81, (byte)50, (byte)99, (byte)57, (byte)80, (byte)81}, byteArray1);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
      assertEquals(4, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Base64 base64_0 = new Base64(42, (byte[]) null);
      base64_0.setInitialBuffer((byte[]) null, (-480), 2036);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64(3570, byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [AQ==]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base64 base64_0 = new Base64();
      boolean boolean0 = base64_0.hasData();
      assertFalse(base64_0.isUrlSafe());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("");
      Base64 base64_0 = new Base64();
      base64_0.setInitialBuffer(byteArray0, 1049, (byte)0);
      boolean boolean0 = base64_0.hasData();
      assertTrue(boolean0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Base64 base64_0 = new Base64();
      int int0 = base64_0.avail();
      assertEquals(0, int0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("efvS$PDjB=HaLS");
      Base64 base64_0 = new Base64();
      // Undeclared exception!
      try { 
        base64_0.decode(byteArray0, 229, 229);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 229
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(1121);
      base64_0.readResults(byteArray0, 2, 1121);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = new byte[1];
      base64_0.setInitialBuffer(byteArray0, (byte)90, 1);
      base64_0.readResults(byteArray0, (byte)90, (byte)90);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom((-918L));
      BigInteger bigInteger0 = new BigInteger(1396, mockRandom0);
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64.encodeBase64Chunked(byteArray0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(1121);
      base64_0.decode(byteArray0);
      base64_0.readResults(byteArray0, 2, 1121);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(1121);
      base64_0.setInitialBuffer(byteArray0, 73, 73);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base64 base64_0 = new Base64(true);
      base64_0.encode(byteArray0);
      base64_0.encode(byteArray0, 9, 9);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("UTF-~16BE");
      Base64 base64_0 = new Base64();
      base64_0.encode(byteArray0, (-1), (-1));
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      Base64.encodeBase64URLSafeString(byteArray0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      Base64.encodeBase64Chunked(byteArray0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      Base64 base64_0 = new Base64(7, byteArray0, false);
      base64_0.encode(byteArray0, (int) (byte)0, 7);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(2);
      base64_0.encodeToString(byteArray0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("efvS$PDjB=HaLS");
      Base64.decodeBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Base64.decodeBase64("WB8");
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      boolean boolean0 = Base64.isBase64((byte) (-108));
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      boolean boolean0 = Base64.isBase64((byte)123);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64.isArrayByteBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      byteArray0[0] = (byte)61;
      byteArray0[1] = (byte)61;
      byteArray0[2] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Base64 base64_0 = new Base64(true);
      Object object0 = base64_0.decode((Object) "(.]]X+HXx *CN t");
      base64_0.encode(object0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      Object object0 = base64_0.decode((Object) "(.]S]X+HX5 *CN t");
      base64_0.decode(object0);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      MockRandom mockRandom0 = new MockRandom((-918L));
      Base64 base64_0 = new Base64();
      try { 
        base64_0.decode((Object) mockRandom0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 decode is not a byte[] or a String
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Base64.decodeBase64((String) null);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = Base64.encodeBase64Chunked((byte[]) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[8];
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, false, false, (int) (byte)0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (14) than the specified maxium size of 0
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      byteArray0[1] = (byte)9;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(4, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0}, byteArray1);
      assertArrayEquals(new byte[] {(byte)0, (byte)9, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(1121);
      byte[] byteArray1 = base64_0.encode(byteArray0);
      byte[] byteArray2 = Base64.discardWhitespace(byteArray1);
      assertArrayEquals(new byte[] {(byte)81, (byte)50, (byte)99, (byte)57, (byte)80, (byte)81, (byte)61, (byte)61, (byte)0, (byte)0}, byteArray2);
      assertEquals(10, byteArray2.length);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[0] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0}, byteArray1);
      assertEquals(3, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)32, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)32;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Base64 base64_0 = new Base64();
      try { 
        base64_0.encode((Object) null);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64((-20), byteArray0);
      String string0 = base64_0.encodeToString((byte[]) null);
      assertNull(string0);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
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
