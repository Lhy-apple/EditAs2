/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:25:37 GMT 2023
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
      BigInteger bigInteger0 = BigInteger.valueOf(2033L);
      byte[] byteArray0 = Base64.encodeInteger(bigInteger0);
      assertEquals(4, byteArray0.length);
      assertArrayEquals(new byte[] {(byte)66, (byte)47, (byte)69, (byte)61}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base64 base64_0 = new Base64(7, byteArray0, false);
      String string0 = base64_0.encodeToString(byteArray0);
      assertEquals("AAAA\u0000\u0000\u0000", string0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      String string0 = Base64.encodeBase64String(byteArray0);
      assertEquals("AAAA\r\n", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      String string0 = Base64.encodeBase64URLSafeString(byteArray0);
      assertEquals("AAAAAAAAAA", string0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65}, byteArray1);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0);
      assertEquals(12, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65, (byte)13, (byte)10}, byteArray1);
      assertEquals(6, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Base64 base64_0 = new Base64(848, (byte[]) null);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64(") than the specified maxium size of ");
      Base64 base64_0 = null;
      try {
        base64_0 = new Base64(75, byteArray0, true);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // lineSeperator must not contain base64 characters: [\uFFFD\u0016\uFFFD\uFFFD\u0017\uFFFD\uFFFD\uFFFD\"~'\uFFFD\uFFFD\uFFFDb\uFFFDk\"\uFFFD\uFFFD\u001F]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Base64 base64_0 = new Base64();
      boolean boolean0 = base64_0.hasData();
      assertFalse(boolean0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = new byte[7];
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertEquals(12, byteArray1.length);
      
      boolean boolean0 = base64_0.hasData();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64(3, byteArray0, true);
      int int0 = base64_0.avail();
      assertTrue(base64_0.isUrlSafe());
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ONE;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      Base64 base64_0 = new Base64();
      // Undeclared exception!
      try { 
        base64_0.decode(byteArray0, 1403, 1403);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 1403
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("Lg");
      Base64 base64_0 = new Base64(1813, byteArray0);
      base64_0.readResults(byteArray0, 0, 0);
      assertFalse(base64_0.isUrlSafe());
      assertArrayEquals(new byte[] {(byte)46}, byteArray0);
      assertEquals(1, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Base64 base64_0 = new Base64((-502));
      byte[] byteArray0 = new byte[7];
      byte[] byteArray1 = base64_0.encode(byteArray0);
      int int0 = base64_0.readResults(byteArray1, 9, (byte) (-47));
      assertFalse(base64_0.isUrlSafe());
      assertEquals((-47), int0);
      assertEquals(12, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      Base64 base64_0 = new Base64((-2815));
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertEquals(4, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)65, (byte)65}, byteArray1);
      
      int int0 = base64_0.readResults(byteArray0, 76, (-1993));
      assertEquals((-1), int0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Base64 base64_0 = new Base64(true);
      base64_0.setInitialBuffer((byte[]) null, 61, 1717986918);
      assertTrue(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64(3, byteArray0, true);
      base64_0.setInitialBuffer(byteArray0, (byte) (-18), (byte) (-128));
      assertTrue(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      byte[] byteArray0 = new byte[7];
      Base64 base64_0 = new Base64();
      base64_0.decode("GxsBxiDGAA");
      base64_0.encode(byteArray0, 111, (int) (byte)27);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64 base64_0 = new Base64((byte)97, byteArray0, true);
      base64_0.encode(byteArray0, (int) (byte)0, (-3332));
      assertTrue(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byte[] byteArray1 = Base64.encodeBase64URLSafe(byteArray0);
      assertEquals(3, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      Base64 base64_0 = new Base64((-1), byteArray0);
      // Undeclared exception!
      try { 
        base64_0.encode(byteArray0, 76, 76);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // 76
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Base64 base64_0 = new Base64(1);
      Object object0 = base64_0.decode((Object) "AA==\r\n");
      Object object1 = base64_0.encode(object0);
      assertFalse(base64_0.isUrlSafe());
      assertNotSame(object1, object0);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Base64 base64_0 = new Base64();
      Object object0 = base64_0.decode((Object) ">vjZ#5l'2IQ-%L>},l0");
      Object object1 = base64_0.decode(object0);
      assertNotSame(object1, object0);
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("L");
      assertEquals(0, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("MArD8OJ");
      assertArrayEquals(new byte[] {(byte)48, (byte)10, (byte) (-61), (byte) (-16), (byte) (-30)}, byteArray0);
      assertEquals(5, byteArray0.length);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      byte[] byteArray0 = Base64.CHUNK_SEPARATOR;
      byte[] byteArray1 = Base64.encodeBase64Chunked(byteArray0);
      boolean boolean0 = Base64.isArrayByteBase64(byteArray1);
      assertEquals(6, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)68, (byte)81, (byte)111, (byte)61, (byte)13, (byte)10}, byteArray1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      Base64 base64_0 = new Base64();
      try { 
        base64_0.decode((Object) bigInteger0);
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
      // Undeclared exception!
      try { 
        Base64.decodeInteger((byte[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      Base64 base64_0 = new Base64(3, byteArray0, true);
      byte[] byteArray1 = base64_0.decode("");
      assertEquals(0, byteArray1.length);
      assertTrue(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      byte[] byteArray0 = Base64.encodeBase64URLSafe((byte[]) null);
      assertNull(byteArray0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      byte[] byteArray0 = new byte[0];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0, false);
      assertEquals(0, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      byte[] byteArray0 = Base64.decodeBase64("org.apache.commons.codec.DecoderException");
      // Undeclared exception!
      try { 
        Base64.encodeBase64(byteArray0, false, false, (-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Input array too big, the output array would be bigger (38) than the specified maxium size of -1
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      byte[] byteArray0 = new byte[9];
      byteArray0[0] = (byte)9;
      byte[] byteArray1 = Base64.discardWhitespace(byteArray0);
      assertEquals(8, byteArray1.length);
      assertArrayEquals(new byte[] {(byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray1);
      assertArrayEquals(new byte[] {(byte)9, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0, (byte)0}, byteArray0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      byte[] byteArray0 = new byte[1];
      byte[] byteArray1 = Base64.encodeBase64(byteArray0, true);
      byte[] byteArray2 = Base64.discardWhitespace(byteArray1);
      assertEquals(6, byteArray2.length);
      assertArrayEquals(new byte[] {(byte)65, (byte)65, (byte)61, (byte)61, (byte)0, (byte)0}, byteArray2);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      byte[] byteArray0 = new byte[4];
      byteArray0[0] = (byte)67;
      byteArray0[1] = (byte)67;
      byteArray0[2] = (byte)67;
      byte[] byteArray1 = Base64.decodeBase64(byteArray0);
      byte[] byteArray2 = Base64.discardWhitespace(byteArray1);
      assertArrayEquals(new byte[] {(byte)8}, byteArray2);
      assertEquals(1, byteArray2.length);
      assertEquals(2, byteArray1.length);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      byte[] byteArray0 = new byte[3];
      byteArray0[0] = (byte)9;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      byte[] byteArray0 = new byte[2];
      byteArray0[0] = (byte)32;
      boolean boolean0 = Base64.isArrayByteBase64(byteArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Base64 base64_0 = new Base64();
      try { 
        base64_0.encode((Object) "org.apache.commons.codec.DecoderException");
        fail("Expecting exception: Exception");
      
      } catch(Exception e) {
         //
         // Parameter supplied to Base64 encode is not a byte[]
         //
         verifyException("org.apache.commons.codec.binary.Base64", e);
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Base64 base64_0 = new Base64();
      byte[] byteArray0 = base64_0.encode((byte[]) null);
      assertNull(byteArray0);
      assertFalse(base64_0.isUrlSafe());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      BigInteger bigInteger0 = BigInteger.ZERO;
      byte[] byteArray0 = Base64.toIntegerBytes(bigInteger0);
      Base64 base64_0 = new Base64();
      byte[] byteArray1 = base64_0.encode(byteArray0);
      assertSame(byteArray1, byteArray0);
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
