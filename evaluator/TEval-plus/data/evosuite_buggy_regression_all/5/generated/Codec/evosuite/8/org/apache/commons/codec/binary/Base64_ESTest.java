/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 22:47:40 GMT 2023
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
      byte[] byteArray0 = new byte[28];
      Base64 base64_0 = new Base64((byte)28, byteArray0);
      base64_0.decode((Object) "Parameter supplied to Base64 encode is not a byte[]");
      base64_0.encode(byteArray0, (-1832), 64);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[108];
      Base64 base64_0 = new Base64((byte)8, byteArray0);
      base64_0.encodeToString(byteArray0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      String string0 = Base64.encodeBase64String(byteArray0);
      assertEquals("Q2c9PQ==", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("I?GeB`p");
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertArrayEquals(new byte[] {(byte)32, (byte)103, (byte) (-127)}, byteArray0);
      assertEquals(3, byteArray0.length);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[47];
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[108];
      BigInteger bigInteger0 = Base64.decodeInteger(byteArray0);
      byte[] byteArray1 = Base64.encodeInteger(bigInteger0);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[28];
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertEquals(38, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      boolean boolean0 = base64_0.isUrlSafe();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Base64 base64_0 = new Base64((-1173), (byte[]) null, false);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64(",Q2syy!\"^RGIq1j>4F");
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64(101, byteArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [Ck2\uFFFD\u0011\uFFFD\uFFFDX\uFFFD]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Base64 base64_0 = new Base64(true);
      boolean boolean0 = base64_0.isUrlSafe();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      Base64 base64_0 = new Base64((byte) (-80), byteArray0, false);
      boolean boolean0 = base64_0.hasData();
      assertFalse(base64_0.isUrlSafe());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      byte[] byteArray0 = new byte[10];
      Base64 base64_0 = new Base64((byte) (-77), byteArray0);
      base64_0.encode(byteArray0, (int) (byte) (-77), (int) (byte) (-77));
      boolean boolean0 = base64_0.hasData();
      assertTrue(boolean0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      Base64 base64_0 = new Base64((byte) (-80), byteArray0, false);
      int int0 = base64_0.avail();
      assertFalse(base64_0.isUrlSafe());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      byte[] byteArray0 = new byte[93];
      Base64 base64_0 = new Base64((byte)8, byteArray0);
      byte[] byteArray1 = new byte[0];
      base64_0.setInitialBuffer(byteArray1, 85, 0);
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray0, 38, (int) (byte)8);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 85
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64();
      base64_0.readResults(byteArray0, (-6088), (-6088));
      assertEquals(4, byteArray0.length);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      byte[] byteArray0 = new byte[74];
      Base64 base64_0 = new Base64((byte)8, byteArray0);
      base64_0.encode(byteArray0, (int) (byte)8, 19);
      int int0 = base64_0.readResults(byteArray0, 19, (byte)8);
      assertFalse(base64_0.isUrlSafe());
      assertEquals(8, int0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[28];
      Base64 base64_0 = new Base64((byte)8, byteArray0);
      String string0 = base64_0.encodeToString(byteArray0);
      assertEquals("AAAAAAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAAAAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAAAAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAAAAAA\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000AAAAAA==\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000\u0000", string0);
      
      int int0 = base64_0.readResults(byteArray0, (byte)8, (byte)8);
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      Base64 base64_0 = new Base64((byte) (-1), byteArray0);
      base64_0.setInitialBuffer((byte[]) null, 1431655765, 176);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[28];
      Base64 base64_0 = new Base64((byte)104, byteArray0);
      base64_0.setInitialBuffer(byteArray0, (byte)104, (byte)104);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.TEN;
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      Base64 base64_0 = new Base64(2819);
      base64_0.encode(byteArray0, 2819, (-891));
      assertEquals(4, byteArray0.length);
      assertArrayEquals(new byte[] {(byte)67, (byte)103, (byte)61, (byte)61}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[28];
      Base64 base64_0 = new Base64((byte)28, byteArray0);
      Object object0 = base64_0.decode((Object) "Parameter supplied to Base64 encode is not a byte[]");
      Object object1 = base64_0.decode(object0);
      assertNotSame(object1, object0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byteArray0[0] = (byte) (-1);
      Base64.decodeBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Base64 base64_0 = new Base64();
      Object object0 = base64_0.decode((Object) "B}y*UW]tISvoa~n+;[U");
      base64_0.encode(object0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("UTF-16");
      Base64.isArrayByteBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = new byte[5];
      Base64 base64_0 = new Base64((byte) (-80), byteArray0, false);
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      base64_0.decode(byteArray1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("Parameter supplied to Base64 decode is not a byte[] or a String");
      Base64.isArrayByteBase64(byteArray0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Base64 base64_0 = new Base64(false);
      try { 
        base64_0.decode((Object) base64_0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 decode is not a byte[] or a String
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[74];
      Base64 base64_0 = new Base64((byte)8, byteArray0);
      byte[] byteArray1 = base64_0.decode((String) null);
      assertNull(byteArray1);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Base64.decodeBase64("");
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      byte[] byteArray0 = Base64.encodeBase64((byte[]) null, false, false);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, false, false, (-155));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (10) than the specified maxium size of -155
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[8] = (byte)9;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray1);
      assertEquals(8, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(2, byteArray1.length);
      assertNotSame(byteArray1, byteArray0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = new byte[6];
      byteArray0[5] = (byte)32;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(5, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byteArray0[0] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Base64 base64_0 = new Base64(true);
      try { 
        base64_0.encode((Object) base64_0);
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      byte[] byteArray0 = new byte[105];
      Base64 base64_0 = new Base64((byte)8, byteArray0);
      String string0 = base64_0.encodeToString((byte[]) null);
      assertNull(string0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = new byte[0];
      String string0 = base64_0.encodeToString(byteArray0);
      assertEquals("", string0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      byte[] byteArray0 = new byte[41];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      BigInteger bigInteger0 = new BigInteger(byteArray1);
      byte[] byteArray2 = Base64.encodeInteger(bigInteger0);
      assertEquals((short)16701, bigInteger0.shortValue());
      assertEquals(76, byteArray2.length);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
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